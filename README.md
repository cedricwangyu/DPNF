# DPNF
This repository contains python implementations for paper *Differentially Private Normalizing Flows for Density Estimation, Data Synthesis, and Variational Inference with Application to Electronic Health Records*. 
Density estimation, variational inference with differentially private normalizing flow are implemented for an analytical regression model and a circuit model for health data. 

## Citation
Circuit model implementation is cited from [the Schiavazzi Lab at the University of Notre Dame](https://github.com/desResLab/supplMatHarrod20/tree/master/models)
and Normalizing Flow (MADE, MAF, RealNVP) implementation is cited from [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 
## Repository Information 

This repository contains the supplementary material to the paper *K.K. Harrod, J.L. Rogers, J.A. Feinstein, A.L. Marsden and D.E. Schiavazzi*, [**Predictive modeling of secondary pulmonary hypertension in left ventricular diastolic dysfunction**](https://www.frontiersin.org/articles/10.3389/fphys.2021.666915/full).

We provide the two datasets used to generate the results in the paper. Additionally we also provide some python scripts to run zero-dimensional hemodynamic models representing simple RC and RCR models, and one model for adult cardiovascular physiology.

## Requirements
* Python 3.10.0
* PyTorch 1.11.0
* Numpy 1.22.3
* Scipy 1.8.0
* Cvsim6

cvsim6 is limited for experiment with circuit model, which is contained in folder [supplMatHarrod20](supplMatHarrod20/models). 
To complile this package, please cd to the folder [models](supplMatHarrod20/models) and execute the following in terminal
```
python3 setup.py build_ext --inplace
```
For additional information, please refer to the [README](supplMatHarrod20/README.md) and the other [README](supplMatHarrod20/models/README.md).

## Density Estimation
To run with EHR data, we recommend
```
python run_experiment.py --task de --name EHR_DE --job ehr --data_dir source/data/EHR_private/EHR.txt --output_dir results/EHR_DE --flow_type maf --n_blocks 15 --hidden_size 1 --input_size 19 --n_iter 8000 --optimizer_type sgd --lr 0.00002 --lr_decay 0.9999 --n_sample 5000 --noisy False --poisson_ratio 0.5
```
To run with regression model, we recommend
```
python run_experiment.py --task de --name REG_DE --job reg --data_dir source/data/Regression_private/eps_2.txt --output_dir results/REG_DE --flow_type maf --n_blocks 18 --hidden_size 100 --input_size 9 --n_iter 8000 --optimizer_type rmsprop --lr 0.002 --lr_decay 0.9995 --n_sample 5000 --noisy False --poisson_ratio 0.08333333333
```
