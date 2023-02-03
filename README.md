# DPNF
This repository contains python implementations for paper *Differentially Private Normalizing Flows for Density Estimation, Data Synthesis, and Variational Inference with Application to Electronic Health Records*. 
Density estimation, variational inference with differentially private normalizing flow are implemented for an analytical regression model and a circuit model for health data. 

## Citation
Circuit model implementation is cited from [the Schiavazzi Lab at the University of Notre Dame](https://github.com/desResLab/supplMatHarrod20/tree/master/models)
and Normalizing Flow (MADE, MAF, RealNVP) implementation is cited from [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 

## Requirements
* Python 3.10.0
* PyTorch 1.11.0
* Numpy 1.22.3
* Scipy 1.8.0
* Cvsim6

cvsim6 is limited for experiment with circuit model, which is contained in folder [supplMatHarrod20](https://github.com/cedricwangyu/DPNF/supplMatHarrod20). 
To complile this package, please cd to the folder [models](https://github.com/cedricwangyu/DPNF/supplMatHarrod20/models) and execute the following in terminal
```
python3 setup.py build_ext --inplace
```
For additional information, please refer to the [README](https://github.com/cedricwangyu/DPNF/supplMatHarrod20/README.md) and the other [README](https://github.com/cedricwangyu/DPNF/supplMatHarrod20/models/README.md).

## Density Estimation
To run with EHR data without differential privacy, we recommend
```
python run_experiment.py --task de --name EHR_DE --job ehr --data_dir source/data/EHR_private/EHR.txt --output_dir results/EHR_DE --flow_type maf --n_blocks 15 --hidden_size 1 --input_size 19 --n_iter 8000 --optimizer_type sgd --lr 0.00002 --lr_decay 0.9999 --n_sample 5000 --noisy False
```
To run with EHR data with differential privacy (sigma = 6.10 for example), we recommend
```
python run_experiment.py --task de --name EHR_DE --job ehr --data_dir source/data/EHR_private/EHR.txt --output_dir results/EHR_DE --flow_type maf --n_blocks 15 --hidden_size 1 --input_size 19 --n_iter 8000 --optimizer_type sgd --lr 0.00002 --lr_decay 0.9999 --n_sample 5000 --noisy True --poisson_ratio 0.5 --sigma 6.10 --C 10
```

To run with regression model without differential privacy, we recommend
```
python run_experiment.py --task de --name REG_DE --job reg --data_dir source/data/Regression_private/eps_2.txt --output_dir results/REG_DE --flow_type maf --n_blocks 18 --hidden_size 100 --input_size 9 --n_iter 8000 --optimizer_type rmsprop --lr 0.002 --lr_decay 0.9995 --n_sample 5000 --noisy False
```
To run with regression model with differential privacy (sigma = 6.68 for example), we recommend
```
python run_experiment.py --task de --name REG_DE --job reg --data_dir source/data/Regression_private/eps_2.txt --output_dir results/REG_DE --flow_type maf --n_blocks 18 --hidden_size 100 --input_size 9 --n_iter 8000 --optimizer_type rmsprop --lr 0.002 --lr_decay 0.9995 --n_sample 5000 --noisy True --poisson_ratio 0.08333333333 --sigma 6.68 --C 10
```
## Variational Inference
To run with regression model without differential privacy, we recommend
```
python run_experiment.py --task vi --job reg --data_dir source/data/data_reg.txt --n_blocks 1 --n_hidden 1 --hidden_size 10 --input_size 5 --n_iter 8000 --batch_size 1000 --lr 0.01
```
To run with regression model with differential privacy (sigma = 6.68 for example), we recommend
```
python run_experiment.py --task vi --job reg --data_dir source/data/data_reg.txt --n_blocks 1 --n_hidden 1 --hidden_size 10 --input_size 5 --n_iter 8000 --batch_size 1000 --lr 0.01 --noisy True --sigma 6.68 --C 10 --poisson_ratio 0.083333333
```

To run with EHR data, the surrogate model should be trained, therefore *circuit_model.npz* and *circuit_model.sur* needs to be presented in root directory. If not, please execute
```
python EHR.py --data_dir source/data/EHR_private/EHR.txt
```
These two files are also contained in folder [trained_circuit_surrogate](https://github.com/cedricwangyu/DPNF/source/data/trained_circuit_surrogate). Then we recommend
```
python run_experiment.py --task vi --job ehr --data_dir source/data/EHR_private/EHR_cvsim6.txt --n_blocks 5 --n_hidden 1 --hidden_size 100 --input_size 2 --n_iter 15000 --batch_size 500 --optimizer_type rmsprop --lr 0.01 --scheduler_order True --lr_decay 0.9999 
```
GPU is supported with setting ```--no_cuda False```. ```compute_sigma.py``` provides functions to compute the size of noise given epsilon or mu.
## Hyper Parameters
We summarize all hyper parameters used in these experiments.

| parameter                 | argument           | EHR DP-DE  | EHR VI     | Reg DP-VI  | Reg DP-DE  | Reg VI     |
|---------------------------|--------------------|------------|------------|------------|------------|------------|
| Task type                 | --task             | de         | vi         | vi         | de         | vi         |
| Choice of experiment      | --job              | ehr        | ehr        | reg        | reg        | reg        |
| Normalizing Flow          | --flow_type        | maf        | maf        | maf        | maf        | maf        |
| Number of blocks          | --block_n          | 15         | 5          | 1          | 18         | 1          |
| Number of hidden layers   | --hidden_n         | 1          | 1          | 1          | 1          | 1          |
| Hidden layer size         | --hidden_size      | 200        | 100        | 10         | 100        | 10         |
| Activation function       | --activation_fn    | relu       | relu       | relu       | relu       | relu       |
| Input order for mask      | --input_order      | sequential | sequential | sequential | sequential | sequential |
| Batch normalization layer | --batch_norm_order | True       | True       | True       | True       | True       |
| Input size                | --input_size       | 19         | 2          | 5          | 9          | 5          |
| Number of iterations      | --n_iter           | 8000       | 15000      | 8000       | 8000       | 8000       |
| Batch size                | --batch_size       | 100        | 500        | 1000       | 100        | 1000       |
| Optimizer                 | --optimizer_type   | sgd        | rmsprop    | rmsprop    | rmsprop    | rmsprop    |
| Scheduler                 | --scheduler_order  | False      | True       | True       | True       | True       |
| Exponential decay factor  | --lr_decay         |            | 0.9999     | 0.999      | 0.9995     | 0.999      |
| Noisy gradient            | --noisy            | True/False | False      | True/False | True/False | False      |
| Clipping constant         | --C                | 10         |            | 10         | 5          |            |
| Poisson sampling rate     | --poisson_ratio    | 0.5        |            | 0.08333333 | 0.08333333 |            |

Note that
* **EHR DP-DE**: Differentially private density estimation on EHR data (Section 3.2).
* **EHR VI**: Variational inference with normalizing flow with circuit model on EHR data (Section 3.4.2).
* **Reg DP-VI**: Differentially private variational inference with normalizing flow on regression model (Section 4.2).
* **Reg DP-DE**: Differentially private density estimation on regression model (Section 4.2).
* **Reg VI**: Variational inference with normalizing on regression model (Section 4.2).







