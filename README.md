# D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion

## Requirements

The main dependencies `torch==1.10.1` and `torch-geometric==2.0.4`. We use CUDA 10.2. Refer to `requirements.txt` for more details.

## Dataset

Download the datasets from [here](https://drive.google.com/drive/folders/1pwmeST3zBcSC34KbAL_Wvi-cFtufAOCE?usp=sharing) to `data/`

## Base GNNs

TODO

## Training

- Train d4_explainer: `python diff_main.py`
- Train baseline explainers: `python baseline_main.py`

### Datasets supported:

Node classification: `BA_shapes`; `Tree_Cycle`; `Tree_Grids`; `cornell`

Graph classification: `mutag`; `ba3`; `bbbp`;`NCI1`

### Optional arguments

Refer to `diff_main.py` and `baseline_main.py` for more details.

## Evaluation

- In-distribution evaluation: `python -m evaluation.in_distribution.ood_evaluation`
- Robustness evaluation: `python -m evaluation.robustness`

### Optional arguments

Refer to `evaluation/in_distribution/ood_evaluation.py` and `evaluation/robustness/robustness_evaluation.py` for more details.
