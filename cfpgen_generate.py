import yaml
import argparse
from pprint import pprint
import torch
import os
from byprot import utils
from byprot.models.lm.cfp_gen import CondDiffusionProteinLanguageModel
import pandas as pd
from omegaconf import DictConfig
import pyrootutils
import pickle
from tqdm import tqdm
import random
import numpy as np
import time
import multiprocessing as mp
from torch.cuda.amp import autocast
import traceback

def load_pkl_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_motif_gt_pos(target, model, motif_start_end, min_mask_ratio=0.05, max_mask_ratio=0.1):
    batch_size, sequence_length = target.shape
    masked_targets = []

    for i in range(batch_size):
        current_target = target[i].clone()

        non_special_sym_mask = (
                (current_target != model.pad_id) &
                (current_target != model.bos_id) &
                (current_target != model.eos_id)
        )
        effective_indices = torch.where(non_special_sym_mask)[0]

        if len(effective_indices) == 0:
            masked_targets.append(torch.full_like(current_target, fill_value=model.mask_id))
            continue

        total_length = len(effective_indices)
        retain_min_len = max(10, int(min_mask_ratio * total_length))
        retain_max_len = max(30, int(max_mask_ratio * total_length))

        start, end = motif_start_end[i]

        if start == 0 and end == 0:
            retain_length = torch.randint(retain_min_len, retain_max_len + 1, (1,)).item()
            retain_start_idx = torch.randint(0, total_length - retain_length + 1, (1,)).item()
            retain_start = effective_indices[retain_start_idx].item()
            retain_end = effective_indices[retain_start_idx + retain_length - 1].item()
        else:
            motif_length = end - start
            if motif_length < retain_min_len:
                retain_length = retain_min_len
            elif motif_length > retain_max_len:
                retain_length = retain_max_len
            else:
                retain_length = motif_length

            if end - start - retain_length > 0:
                retain_start = torch.randint(start, end - retain_length + 1, (1,)).item()
            else:
                retain_start = start

            retain_end = retain_start + retain_length - 1

        sequence_indices = torch.arange(sequence_length, device=target.device)
        mask = non_special_sym_mask & ((sequence_indices < retain_start) | (sequence_indices > retain_end))
        masked_target = current_target.clone()
        masked_target[mask] = model.mask_id

        masked_targets.append(masked_target)

    return torch.stack(masked_targets)


def load_label_mappings(vocab_dir=None):
    if vocab_dir is None:
        return None, None, None

    vocab_dir = str(vocab_dir)

    go_map_path = os.path.join(vocab_dir, "go_mapping.pkl")
    ipr_map_path = os.path.join(vocab_dir, "ipr_mapping.pkl")
    ec_map_path = os.path.join(vocab_dir, "ec_mapping.pkl")

    go_mapping = load_pkl_file(go_map_path) if os.path.isfile(go_map_path) else None
    ipr_mapping = load_pkl_file(ipr_map_path) if os.path.isfile(ipr_map_path) else None
    ec_mapping = load_pkl_file(ec_map_path) if os.path.isfile(ec_map_path) else None

    # dag = _get_ontology_dag(config)
    return go_mapping, ipr_mapping, ec_mapping

def expand_ec_wildcards(ec_annos, ec_mapping, max_per_wildcard=1):
    if not ec_annos or ec_mapping is None:
        return []

    expanded = []
    seen = set()

    for raw in ec_annos:
        ec = raw.strip()
        if not ec:
            continue

        if ec in ec_mapping:
            if ec not in seen:
                seen.add(ec)
                expanded.append(ec)
            continue


        parts = ec.replace("-", "").rstrip(".").split(".")
        parts = [p for p in parts if p]
        if not parts:
            continue
        prefix = ".".join(parts) + "."

        candidates = [key for key in ec_mapping if key.startswith(prefix)]

        if not candidates:
            print(f"[EC wildcard] No match in ec_mapping for '{raw}' (prefix='{prefix}')")
            continue

        if len(candidates) > max_per_wildcard:
            candidates = random.sample(candidates, max_per_wildcard)

        for key in candidates:
            if key not in seen:
                seen.add(key)
                expanded.append(key)

        print(f"[EC wildcard] '{raw}' → {len(candidates)} sampled from {prefix}* : {candidates}")

    return expanded



def get_initial(config, model, sample, length, tokenizer, device, sequence):

    # go_labels = sample['go_f_mapped'] if 'go_f_mapped' in sample else sample['go_mapped']
    # ipr_labels = sample['ipr_mapped']
    # ec_labels = sample.get('EC_mapped', None)

    go_mapping, ipr_mapping, ec_mapping = load_label_mappings(config.get('vocab_dir', None))

    if config.get("use_go", False):
        go_labels = []
        go_raw = sample.get("go_numbers")
        go_annos = (
            [go for v in (go_raw or {}).values() for go in (v or [])]
            if isinstance(go_raw, dict)
            else list(go_raw or [])
        )
        if go_mapping is not None and len(go_annos) > 0:
            go_labels = [go_mapping[x] for x in go_annos if x in go_mapping]

            # go_terms_all = []
            # global_seen = set()
            # for t in go_annos:
            #     for anc in all_parents_go(t, go_dag, include_self=True):
            #         if anc not in global_seen:
            #             global_seen.add(anc)
            #             go_terms_all.append(anc)
            # go_labels = [go_mapping[x] for x in go_terms_all if x in go_mapping]

    if config.get("use_ipr", False):
        ipr_labels = []
        ipr_annos = sample.get("ipr_numbers")
        if ipr_mapping is not None and len(ipr_annos) > 0:
            ipr_labels = [ipr_mapping[x] for x in ipr_annos if x in ipr_mapping]

    if config.get("use_ec", False):
        ec_labels = []
        ec_annos = sample.get("EC_number")
        if ec_mapping is not None and len(ec_annos) > 0:
            expanded_ec = expand_ec_wildcards(ec_annos, ec_mapping)
            ec_labels = [ec_mapping[x] for x in expanded_ec]

    if config.get('use_seq_motif', False):
        motif_info = sample.get('motif', [])
        if motif_info:
            motif_start_end = [[motif['start'], motif['end']] for motif in motif_info][0]
        else:
            motif_start_end = [0, 0]
        length = len(sequence)

    seq = ['<mask>'] * length

    seq = [''.join(seq)]
    init_seq = seq * 1 #config['num_seqs']
    batch = tokenizer.batch_encode_plus(init_seq,
                                        add_special_tokens=True,
                                        padding="longest",
                                        return_tensors='pt')

    if config.get('use_seq_motif', False):
        seq_cond = tokenizer.batch_encode_plus([sequence],
                                               add_special_tokens=True,
                                               padding="longest",
                                               return_tensors='pt')['input_ids']
        seq_cond = get_motif_gt_pos(seq_cond, model, torch.tensor(motif_start_end).unsqueeze(0))


    out_batch = {
        'input_ids': batch['input_ids'],
        'input_mask': batch['attention_mask'].bool(),
    }

    # Annotation tags
    if config.get('use_go', False) and len(go_labels):
        out_batch['go_label'] = torch.tensor(go_labels)

    if config.get('use_ipr', False) and len(ipr_labels):
        out_batch['ipr_label'] = torch.tensor(ipr_labels)

    if config.get('use_ec', False) and len(ec_labels):
        out_batch['ec_label'] = torch.tensor(ec_labels)

    if config.get('use_seq_motif', False):
        out_batch['seq_cond'] = seq_cond

    return utils.recursive_to(out_batch, device)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_data_sequentially(data, num_splits):
    split_size = len(data) // num_splits
    remainder = len(data) % num_splits

    splits = []
    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        splits.append(data[start_idx:end_idx])
        start_idx = end_idx

    return splits


def process_on_gpu(gpu_idx, part_data, config, part_fasta_filename):
    try:
        print(f"Starting processing on GPU {gpu_idx} with {len(part_data)} sequences...")

        model = CondDiffusionProteinLanguageModel.from_pretrained(config['ckpt_path'])
        model = model.eval().cuda(gpu_idx)
        tokenizer = model.tokenizer

        set_seed(config.get('seed', 42) + gpu_idx)

        with open(part_fasta_filename, 'a') as fp_save:
            for index, row in enumerate(part_data):

                sequence = row['sequence']
                seq_id = row['uniprot_id']

                print(f"Generating for protein {seq_id} on GPU {gpu_idx}:")

                seq_len = random.randint(config['seq_lens'][0], config['seq_lens'][1])
                device = torch.device(f"cuda:{gpu_idx}")
                batch = get_initial(config, model, row, seq_len, tokenizer, device, sequence)

                partial_mask = batch['input_ids'].ne(model.mask_id).type_as(batch['input_mask'])

                with autocast():
                    outputs = model.generate(batch=batch, tokenizer=tokenizer,
                                             max_iter=config['max_iter'],
                                             sampling_strategy=config['sampling_strategy'],
                                             partial_masks=partial_mask)
                output_tokens = outputs[0]
                output_results = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

                for _, seq in enumerate(output_results):
                    seq = seq.replace(" ", "")
                    fp_save.write(f">SEQUENCE_ID={seq_id}_L={seq_len}\n")
                    fp_save.write(f"{seq}\n")

        print(f"Finished processing on GPU {gpu_idx}.")

    except Exception as e:
        print(f"Error occurred on GPU {gpu_idx}: {e}")
        traceback.print_exc()


def main(config):

    # multi-process
    mp.set_start_method('spawn', force=True)

    with open(config['input_data'], 'rb') as f:
        input_data = pickle.load(f)

    # detect multi-gpu
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        raise RuntimeError("No GPU devices found!")

    print(f"Detected {gpu_num} GPUs.")

    os.makedirs(config['saveto'], exist_ok=True)
    basename = os.path.basename(os.path.dirname(os.path.dirname(config['ckpt_path'])))

    start_time = time.time()

    if gpu_num == 1:
        final_fasta_filename = os.path.join(config['saveto'], f"{basename}_{config['run_name']}.fasta")
        process_on_gpu(0, input_data, config, final_fasta_filename)
    else:
        part_filenames = [
            os.path.join(config['saveto'], f"{basename}_{config['run_name']}_part_{i}.fasta") for i in range(gpu_num)
        ]
        data_parts = split_data_sequentially(input_data, gpu_num)
        processes = []
        for gpu_idx, part_data in enumerate(data_parts):
            p = mp.Process(target=process_on_gpu, args=(gpu_idx, part_data, config, part_filenames[gpu_idx]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total computation time: {hours} hours, {minutes} minutes, {seconds} seconds")

    if gpu_num > 1:
        final_fasta_filename = os.path.join(config['saveto'], f"{basename}_{config['run_name']}.fasta")
        with open(final_fasta_filename, 'w') as final_fp:
            for part_fasta_filename in part_filenames:
                with open(part_fasta_filename, 'r') as part_fp:
                    final_fp.write(part_fp.read())

        print(f"All parts have been merged into {final_fasta_filename}")


if __name__ == '__main__':
    config_path = 'configs/test_cfpgen.yaml'
    config = load_config(config_path)
    main(config)
