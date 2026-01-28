#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from src.byprot.utils.ontology import Ontology
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import re
import random

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--test-predictions', '-tp', default='test_preds_mf.tsv',
    help='Test data set name')
def main(data_root, ont, test_predictions):

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    obo_path = os.path.join(base_dir, 'data-bin', 'go.obo')
    ontology = Ontology(obo_path, with_rels=True)
    def load_pkl_file(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    test_data = load_pkl_file(data_root)

    predictions = {}
    use_name = False
    with open(test_predictions) as f:
        for line in f:
            it = line.strip().split('\t')
            # import pdb;pdb.set_trace()
            if 'prompt_first_seq30' in it[0]:
                prot_id = re.match(r'prompt_first_seq30_([\w\d]+)', it[0]).groups()[0]
            elif 'name=' in it[0]:
                prot_id = re.match(r'name=([\w\d\.]+)', it[0]).groups()[0]
                use_name = True
            elif 'recovery' in it[0]:
                prot_id = re.match(r'([\w\d]+)', it[0]).groups()[0]
            elif 'SEQUENCE_ID=' in it[0]:
                prot_id = re.match(r'SEQUENCE_ID=([\w\d\.]+)_L', it[0]).groups()[0]
            elif 'SEQUENCE_' in it[0]:
                prot_id = re.match(r'SEQUENCE_([\w\d\.]+)_L=', it[0]).groups()[0]
            elif '_seq30_' in it[0]:
                prot_id = re.match(r'go_prompt_longest_motif_seq30_([\w\d\.]+)', it[0]).groups()[0]
            else :
                prot_id = it[0]
            go_id = it[1]
            score = float(it[2])
            if prot_id not in predictions:
                predictions[prot_id] = {}
            predictions[prot_id][go_id] = score

    preds = {k:list(v.keys()) for k,v in predictions.items()}

    if use_name:
        gts = {ele['name']:set(ele['go_numbers']['F']) for ele in test_data}
    else:
        gts = {ele['uniprot_id']:set(ele['go_numbers']['F']) for ele in test_data}

    gt_list = [gts[uid] for uid in preds.keys()]
    # gt_list = list(gts.values())  # for uncond

    for i, this_gt_go in enumerate(gt_list):
        new_this_go = []
        for go in this_gt_go:
            new_this_go.extend(ontology.get_ancestors(go))
        gt_list[i] = set(new_this_go)

    unique_go_gt = set()
    for go_set in gt_list:
        unique_go_gt.update(go_set)

    pred_list = list(preds.values())

    unique_go_pred = set()
    for go_set in pred_list:
        unique_go_pred.update(go_set)

    unique_go = unique_go_gt & unique_go_pred

    for i, this_gt_go in enumerate(pred_list):
        pred_list[i] = set([ele for ele in this_gt_go if ele in unique_go])
    for i, this_gt_go in enumerate(gt_list):
        gt_list[i] = set([ele for ele in this_gt_go if ele in unique_go])

    mlb = MultiLabelBinarizer()
    all_go_terms = gt_list + pred_list
    mlb.fit(all_go_terms)

    # transform to multi-hot
    y_true_binary = mlb.transform(gt_list)
    y_pred_binary = mlb.transform(pred_list)
    precision_mac = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    recall_mac = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    f1_mac = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)

    precision_mic = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    recall_mic = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    f1_mic = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)


    # AUC-ROC
    auc_roc_macro = roc_auc_score(y_true_binary, y_pred_binary, average='macro')
    auc_roc_micro = roc_auc_score(y_true_binary, y_pred_binary, average='micro')

    # AUC-PR (AUPR)
    aupr_macro = average_precision_score(y_true_binary, y_pred_binary, average='macro')
    aupr_micro = average_precision_score(y_true_binary, y_pred_binary, average='micro')

    output_log = test_predictions.replace('.tsv', '_go-eval.log')


    print(f'F1 Score (Micro): {f1_mic:.3f}')
    print(f'F1 Score (Macro): {f1_mac:.3f}')
    print(f'AUPR (Macro): {aupr_macro:.3f}')
    print(f'AUC-ROC (Macro): {auc_roc_macro:.3f}\n')

    print(f'AUPR (Micro): {aupr_micro:.3f}')
    print(f'AUC-ROC (Micro): {auc_roc_micro:.3f}\n')

    print(f'Precision (Macro): {precision_mac:.3f}')
    print(f'Recall (Macro): {recall_mac:.3f}')
    print(f'Precision (Micro): {precision_mic:.3f}')
    print(f'Recall (Micro): {recall_mic:.3f}')


    with open(output_log, 'w') as log_file:
        log_file.write(f'Precision (Macro): {precision_mac:.4f}\n')
        log_file.write(f'Recall (Macro): {recall_mac:.4f}\n')
        log_file.write(f'F1 Score (Macro): {f1_mac:.4f}\n')
        log_file.write(f'AUC-ROC (Macro): {auc_roc_macro:.4f}\n')
        log_file.write(f'AUPR (Macro): {aupr_macro:.4f}\n\n')

        log_file.write(f'Precision (Micro): {precision_mic:.4f}\n')
        log_file.write(f'Recall (Micro): {recall_mic:.4f}\n')
        log_file.write(f'F1 Score (Micro): {f1_mic:.4f}\n')
        log_file.write(f'AUC-ROC (Micro): {auc_roc_micro:.4f}\n')
        log_file.write(f'AUPR (Micro): {aupr_micro:.4f}\n\n')


if __name__ == '__main__':
    main()
