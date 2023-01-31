



python run_experiment.py --task de --name EHR_DE --job ehr --data_dir source/data/EHR_private/EHR.txt --output_dir results/EHR_DE --flow_type maf --n_blocks 15 --hidden_size 1 --input_size 19 --n_iter 8000 --optimizer_type sgd --lr 0.00002 --lr_decay 0.9999 --n_sample 5000 --noisy False --poisson_ratio 0.5






python run_experiment.py --task de --name REG_DE --job reg --data_dir source/data/Regression_private/eps_2.txt --output_dir results/REG_DE --flow_type maf --n_blocks 18 --hidden_size 100 --input_size 9 --n_iter 8000 --optimizer_type rmsprop --lr 0.002 --lr_decay 0.9995 --n_sample 5000 --noisy False --poisson_ratio 0.08333333333


