<!-- <div align="center"> -->
<!-- omit in toc -->
# _CFP-Gen_: Combinatorial Functional Protein Generation via Diffusion Language Models
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
This repository provides the official PyTorch implementation of our ICML 2025 paper: [_CFP-Gen_: Combinatorial Functional Protein Generation via Diffusion Language Models](https://arxiv.org/pdf/2505.22869). CFP-Gen enables _de novo_ protein design by simultaneously conditioning on functional annotations, sequence motifs, and backbone structures.

## Abstract 

Existing PLMs generate protein sequences based on a single-condition constraint from a specific modality, struggling to simultaneously satisfy multiple constraints across different modalities. In this work, we introduce CFP-Gen, a novel diffusion language model for Combinatorial Functional Protein GENeration. CFP-Gen facilitates the de novo protein design by integrating multimodal conditions with functional, sequence, and structural constraints. Specifically, an Annotation-Guided Feature Modulation (AGFM) module is introduced to dynamically adjust the protein feature distribution based on composable functional annotations, e.g., GO terms, IPR domains and EC numbers. Meanwhile, the ResidueControlled Functional Encoding (RCFE) module captures residue-wise interaction to ensure more precise control. Additionally, off-the-shelf 3D structure encoders can be seamlessly integrated to impose geometric constraints. We demonstrate that CFP-Gen enables high-throughput generation of novel proteins with functionality comparable to natural proteins, while achieving a high success rate in designing multifunctional proteins.

## Features 🌟
![CFP-Gen](./assets/cfpgen.jpg)

CFP-Gen is built upon [DPLM](https://github.com/bytedance/dplm.git).
Beyond the existing capabilities (e.g., unconditional generation, inverse folding, and representation learning),
CFP-Gen focuses on functional protein design and supports the following tasks:
- **Functional Protein Generation**:
  CFP-Gen takes as input functional labels such as Gene Ontology (GO) terms, InterPro (IPR) domains, and Enzyme Commission (EC) numbers, or alternatively uses functional sequence motifs as guidance. CFP-Gen ensures that the generated proteins exhibit functional scores comparable to those of natural proteins.
- **Functional Inverse Folding**:
  We extend traditional inverse folding to _functional inverse folding_, where the goal is not only to design sequences that fold into a given backbone structure, but also to ensure that the resulting proteins exhibit high functional scores. Our released zero-shot version achieves over 10% improvement in F-max compared to ProteinMPNN when conditioned on backbone and annotation inputs.
- **Multifunctional Protein Design**:
  Thanks to its composable function conditioning mechanism, CFP-Gen supports open-ended protein design, where multiple functional constraints (e.g., stacked GO terms, EC numbers, sequence motifs, and structural backbones) can be jointly specified. In such cases, appropriate filtering may be necessary to identify high-quality proteins that meet all specified requirements.

  
  
## Update 
- ​**​[2025-05]​**​ Code of CFP-Gen is released. 



## Installation

```bash
# clone project
git clone --recursive https://github.com/yinjunbo/cfpgen.git
cd cfpgen

# create conda virtual environment
env_name=cfpgen

conda create -n ${env_name} python=3.9 pip
conda activate ${env_name}

# automatically install everything else
bash scripts/install.sh
```


## Datasets
**General Protein Dataset**: The processed dataset ```cfpgen_general_dataset``` can be downloaded from [Google Drive](https://drive.google.com/file/d/1bRtil483NBOuazPSVO7gCpM-K7rNj1Z9/view?usp=sharing) and placed in the directory: ```data-bin/uniprotKB/cfpgen_general_dataset```. It contains 103,939 proteins annotated with 375 GO terms and 1,154 IPR domains.

Files with the suffix ```*_bb.pkl``` additionally include backbone coordinate information. 
The corresponding preprocessed structure files ```uniprot_bb_coords``` can be [downloaded here](https://drive.google.com/file/d/1VXkSd044aJFn-p1dCskMxGjBlLH5mOx7/view?usp=sharing) and placed in: ```data-bin/uniprotKB/uniprot_bb_coords```.


Users may also customize their own datasets for training or fine-tuning by using the provided notebook: ```scripts/create_dataset.ipynb```. 


**Enzyme Dataset**: To be released soon.


## Pretrained Models
We provide several pretrained checkpoints with basic functional capabilities.
More advanced functionalities will be released in future updates.

| Model name                                                                                         | Functionalities       |   
|----------------------------------------------------------------------------------------------------|-----------------------|
| [cfpgen-650m](https://drive.google.com/file/d/1XVYYvKjgcM08v6uI6PYb0-yMzAOxJXis/view?usp=sharing)  | GO & IPR & Seq. Motif |
| [cfpgen-650m-enzyme](https://drive.google.com/file/d/1XVYYvKjgcM08v6uI6PYb0-yMzAOxJXis/view?usp=sharing)  | EC & Seq. Motif |
| [cfpgen-if-zs](https://drive.google.com/file/d/1YwD7xpQTA0ktUdQbgHVb15IKwjEYs2ef/view?usp=sharing) | GO & IPR & Backbone   |
| [dplm-650m](https://drive.google.com/file/d/16_spXxWXAs6E4SWXlCgRqxy6Z7gVZYFV/view?usp=sharing)    | For training CFP-Gen  |



### Notes:
  

- ```cfpgen-650m```: Support conditioning on GO terms, IPR domains and sequence motifs (e.g., 10-30 residue fragments) defined by our **general protein dataset**. This model can be readily used for _Functional Protein Generation_.

- ```cfpgen-650m-enzyme```: Support conditioning on EC numbers (optiioal with GO/IPR) and sequence motifs (e.g., 10-30 residue fragments). To run this model, you need to use the code from the <EC> branch. Note that annotation mappings are also updated in 'cfpgen_650m_enzyme/vocab_with_ec'. 

- ```cfpgen-if-zs```: Designed for _Functional Inverse Folding_ in **zero-shot** settings. The structure adapter used in this model is pretrained on CATH-4.3. It enables the generation of functional sequences conditioned on backbone atomic coordinates, while simultaneously leveraging GO and IPR annotations.

- ```dplm-650m```: This is the base pretrained model from DPLM, required to be placed under ```cfpgen/pretrained/```. It is intended for users who wish to retrain CFP-Gen on their own datasets to support customized functional constraints.


## Generation with _CFP-Gen_
### Functional Protein Generation

Users could modify necessary parameters (e.g.,```ckpt_path=<path_to_cfpgen-650m>```) in the config file:
```bash
configs/test_cfpgen.yaml
```
and then run the following command to start generation:

```bash
python cfp_generate.py
```
The results will be saved in `./generation-results`. Currently, ```cfp_generate.py``` supports generation following ground-truth labels from natural proteins (e.g., from SwissProt).
The [GO/IPR mapping info](https://drive.google.com/drive/folders/1Z6Zmjy1h41rk_Lu89itHeWdep-S9-Zjy?usp=sharing) can be download here. A more flexible interface for open-ended protein design will be released in future updates.

### Functional Protein Inverse Folding

When protein backbone structures are available, users can perform functional inverse folding using the following command:
```bash
python test.py experiment=cfpgen/cfpgen-if_650m_stage2 \
    experiment_path=byprot-checkpoints/cfpgen-if_650m_zero_shot \
    data_split=test \
    ckpt_path=cfpgen-if-zs.ckpt \
    mode=predict \
    ++task.generator.max_iter=100
```
The intial AAR evaluation and corresponding FASTA file will be saved in: ```byprot-checkpoints/cfpgen-if_650m_zero_shot``` or in the directory specified by the ```experiment_path``` argument.

### Unconditional Protein Generation

To facilitate comparison, we also provide proteins generated by DPLM without any conditional input, as a reference baseline:
```bash
python dplm_generate.py
```
The resulting FASTA files will be saved in: ```generation-results/dplm_650m/```.


## Training of _CFP-Gen_

The fully featured version of _CFP-Gen_ requires a multi-stage training process. Each stage typically requires at least 4 A100 GPUs.
Below, we outline key steps for each stage. Users do not need to complete the entire pipeline, and each stage can be trained independently based on the specific functionalities.

### To Support GO/IPR/EC Annotations

Users can first check the parameters in ```configs/experiment/cfpgen/cfpgen_650m_stage1.yaml```, and then run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

max_tokens=8192
accumulate_grad_batches=32

exp=cfpgen/cfpgen_650m_stage1
model_name=cfpgen_general_dataset_stage1

python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 
```
Once training is finished, users will find the logs and checkpoints in
```byprot-checkpoints/${model_name}```.
These checkpoint files can be used for direct evaluation or for second-stage training.

### To Support Functional Motifs

This stage is used to further train the Residue-Controlled Functional Encoding (RCFE) module in a sequence-level controller style.
Users should first obtain the pretrained model from the previous stage by running:
```bash
python scripts/modify_ckpt.py --mode RCFE
````
This will output a checkpoint to the ```pretrained``` directory.
Users should then specify the path to this file via the ```net.pretrained_model_name_or_path``` field in
```configs/experiment/cfpgen/cfpgen_650m_stage2.yaml```.
After that, training can be started with the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

max_tokens=4096
accumulate_grad_batches=32

exp=cfpgen/cfpgen_650m_stage2
model_name=cfpgen_general_dataset_stage2

python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 
```
These checkpoint files can be found in ```byprot-checkpoints/${model_name}``` for evaluation.


### To Support Functional Inverse Folding

**Pretraining of the structure adapter on CATH**

The structure modality requires a pretrained GVP-Transformer and a new structure adapter.
Following DPLM, we first pretrain the adapter on CATH-4.3 (Instructions for preparing the CATH-4.3 dataset can be found in [DPLM](https://github.com/bytedance/dplm/)) using the following command.


```bash
exp=cfpgen/cfpgen-if_650m_stage1
dataset=cath_4.3
name=cfpgen_650m_cath43_stage1

python train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
````

**Functional inverse folind with the _Zero-Shot_ model**

Assuming that we have obtained:
- a pretrained model supporting functional annotations (e.g., ```byprot-checkpoints/cfpgen_general_dataset_stage1```), and
- a model with a structure adapter pretrained on CATH-4.3 (e.g., ```byprot-checkpoints/cfpgen_650m_cath43_stage1```)

We can directly combine the two models into a _zero-shot model_ (no need for further fine-tuning), 
to enable functional inverse folding using the following command:

```bash
python scripts/modify_ckpt.py --mode ZS
```
The combined checkpoint is saved at ```pretrained/cfpgen_650m_cath43.ckpt```. This model can be used with ```configs/experiment/cfpgen/cfpgen-if_650m_stage2.yaml``` (See [Functional Protein Inverse Folding](#functional-protein-inverse-folding)).


**Functional inverse folind with the _SFT_ model**

To ensure better performance on user-specific datasets, it is encouraged to perform supervised fine-tuning.
First, obtain the pretrained model from the zero-shot version by running:
```bash
python scripts/modify_ckpt.py  --mode SFT
```
Then, using this command to start the fine-tuning:
```bash
exp=cfpgen/cfpgen-if_650m_stage2
name=cfpgen-if_650m_stage2

python train.py \
    experiment=${exp} name=${name} \
````
Users can modify the dataset loader registered as ```@register_datamodule('uniprotKB_if')``` to adapt it to their own dataset with backbone structure inputs.




# Evaluation
The evaluation metrics are provided across three levels:
- Sequence level (e.g., distributional statistics)
- Function level (e.g., evaluated by function predictors)
- Structural level (e.g., for inverse folding tasks)

### Distribution Evaluation
The following command computes Maximum Mean Discrepancy (MMD) and Mean Reciprocal Rank (MRR) between the generated and real sequences:
```bash
python eval_mmd.py <go/ipr/ec> <fasta_filename> <gt_data>
```
Here, ```<fasta_filename>```is the output FASTA file obtained by the previous generation commands. ```<gt_data>``` refers to the ground-truth data file (e.g., ```data-bin/uniprotKB/cfpgen_general_dataset/test.pkl```)


### GO Function Evaluation
Users should first set up the environment and prepare the GO database ([data.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/data.tar.gz)) following the instructions from [DeepGO](https://github.com/bio-ontology-research-group/deepgo2), and then run:
```bash
python predict.py -if <fasta_filename>
```
This will output the predicted GO annotation file named: ```<fasta_name>_mf.tsv```. To compute classification scores, use the following command:
```bash
python eval_go.py \
    -dr <gt_data> \
    -tp `<fasta_name>_mf.tsv
```

### IPR Function Evaluation
We apply [InterProScan](https://interproscan-docs.readthedocs.io/en/v5/) to get the IPR domain annotations:
```bash
python eval_ipr.py <fasta_filename> <gt_data>
```
This command will also output the classification scores once finishing predicting the IPR labels.
Please note that running InterProScan in high-throughput mode can be extremely time-consuming.
We recommend executing each step in ```eval_ipr.py``` carefully to ensure that the intermediate outputs are correct before proceeding to the next step.


### EC Function Evaluation
Users need to set up the environment following [CLEAN](https://github.com/tttianhao/CLEAN), and then run:
```bash
python CLEAN_infer_fasta.py --fasta_data <fasta_filename> 
```
This will output the predicted EC annotations at: ```<fasta_name>_maxsep.csv```. To get the classification scores, one can run:
```bash
python eval_ec.py <fasta_name>_maxsep.csv <gt_data>
```

### Inverse Folind Evaluation
Once users obtain the designed sequences (either from the zero-shot model or the SFT model),
they can run the following script to evaluate structure self-consistency, including pLDDT and scTM scores between the designed protein and the corresponding real structure:
```bash
bash eval_struc.sh
```
This script requires a properly configured [esmfold](https://github.com/facebookresearch/esm/) environment, and users must also specify the path to```TMalign_cpp```in the```run_tmscore()```function.



# Acknowledgements

Our project is partially supported by the following open-source codebases. We sincerely thank the original authors for their valuable contributions.

* [DPLM](https://github.com/bytedance/dplm/)
* [ESM3](https://github.com/facebookresearch/esm/)
* [DiT](https://github.com/facebookresearch/DiT)
* [DeepGO](https://github.com/bio-ontology-research-group/deepgo2)
* [ProteoGAN](https://github.com/timkucera/proteogan)

# Citation

```
@inproceedings{yin2025cfpgen,
  title={CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models},
  author={Yin, Junbo and Zha, Chao and He, Wenjia and Xu, Chencheng and Gao, Xin},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
