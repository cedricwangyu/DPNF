import argparse

def parse(des='DP_Normalizing_Flow'):
    parser = argparse.ArgumentParser(description=des)

    # Task Parameters
    parser.add_argument('--name', type=str, default='DP-NF',
                        help='name of the task')
    parser.add_argument('--task', type=str, default='vi', choices=['vi', 'de'],
                        help='type of task: Variational Inference or Density estimation')
    parser.add_argument('--job', type=str, default='reg', choices=['reg', 'ehr'],
                        help='the choice of experiment in paper')
    parser.add_argument('--log_file', type=str, default='log.txt',
                        help='the name of log file including extension')
    parser.add_argument('--output_dir', type=str, default='./results/DP-NF',
                        help='output directory')
    parser.add_argument('--data_dir', type=str, default="source/data/Regression_private/eps_50.txt",
                        help='data directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how often to show loss stat')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed used, default to be random')
    parser.add_argument('--no_cuda', type=str, default='True', choices=['True', 'False'],
                        help='avoid using GPU')

    # NF Parameters
    parser.add_argument('--flow_type', type=str, default='maf', choices=['maf', 'realnvp'],
                        help='type of normalizing flow')
    parser.add_argument('--n_blocks', type=int, default=1,
                        help='the number of blocks used in NF')
    parser.add_argument('--n_hidden', type=int, default=1,
                        help='the number of layers for neural network in blocks of NF')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='the number of nodes in each hidden layer in neural network in blocks of NF')
    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'tanh'],
                        help='activation function used for blocks of NF')
    parser.add_argument('--input_order', type=str, default='sequential', choices=['sequential', 'random'],
                        help='input order for create_mask function of MAF/RealNVP')
    parser.add_argument('--batch_norm_order', type=str, default='True', choices=['True', 'False'],
                        help='order to decide if batch_norm is used')

    # Model Parameters
    parser.add_argument('--input_size', type=int, default=5, choices=range(1, 10001),
                        help='dimensionality of input')
    parser.add_argument('--n_iter', type=int, default=8000,
                        help='number of iterations')
    parser.add_argument('--n_sample', type=int, default=5000,
                        help='the number of samples that will be stored to track parameter cloud')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='number of samples generated for Monte Carlo procedure in variational inference')

    # Optimizer Parameters
    parser.add_argument('--optimizer_type', type=str, default='rmsprop', choices=['sgd', 'rmsprop'],
                        help='optimizer for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate of optimizer')
    parser.add_argument('--scheduler_order', type=str, default='True', choices=['True', 'False'],
                        help='if an exponential scheduler will be used')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='the decay factor for scheduler')

    # DP parameters
    parser.add_argument('--noisy', type=str, default='False', choices=['True', 'False'],
                        help='order to inject noise into gradient')
    parser.add_argument('--C', type=float, default=1.0,
                        help='clipping constant for gradient')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='standard deviation of Gaussian noise that will be injected into gradient')
    parser.add_argument('--poisson_ratio', type=float, default=0.1,
                        help='ratio of Poisson subsampling')

    # parser.add_argument('--', type=, default=, help='')
    args = parser.parse_args()
    return args