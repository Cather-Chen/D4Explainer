# D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion


## Requirements

* `pytorch > 1.8`, `pyg = 2.0.4`

* Refer to `requirements.txt` for more details.

## Dataset
Download the datasets from [here](https://drive.google.com/drive/folders/1pwmeST3zBcSC34KbAL_Wvi-cFtufAOCE?usp=sharing) to `data/`

## Base GNNs
TODO



## Training
* train d4_explainer: `python diff_main.py`
* train baseline explainers: `python baseline_main.py`

### Datasets supported:
node classification: `BA_shapes`; `Tree_Cycle`; `Tree_Grids`; `cornell`

graph classification: `mutag`; `ba3`; `bbbp`;`NCI1`


### Optional arguments



## Evaluation
### In-distirbution Evaluation
### Robustness Evaluation