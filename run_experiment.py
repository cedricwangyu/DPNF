import os
import random
import torch
import numpy as np
from maf import MAF, RealNVP
from utility import parse

args = parse('DP_Normalizing_Flow')
torch.set_default_tensor_type(torch.DoubleTensor)


def execute_DE(args, device, model, optimizer, scheduler):
    from train_DE import train, train_noisy

    if args.job == "reg":
        from RegressionModels import Regression
        rt = Regression(device)
        rt.NI = 6000
        rt.sigma0 = 0.2
        rt.sigma = 0.2
        data = torch.cat([rt.X, rt.Y], dim=1)
    elif args.job == "ehr":
        data = (torch.Tensor(np.loadtxt(args.data_dir))[:64, ]).to(device)

    loglist = []

    if args.noisy == 'False':
        for i in range(args.n_iter + 1):
            if scheduler:
                scheduler.step()
            ind = (torch.Tensor(data.size(0)).uniform_(0, 1) < args.poisson_ratio).to(device)
            train(model, data[ind, :], optimizer, i, args, loglist, device=device)
    else:
        for i in range(args.n_iter + 1):
            if scheduler:
                scheduler.step()
            ind = (torch.Tensor(data.size(0)).uniform_(0, 1) < args.poisson_ratio).to(device)
            train_noisy(model, data[ind, :], optimizer, i, args, loglist, device=device)

    torch.save(model.state_dict(), args.output_dir + '/MAF_params')
    np.savetxt(args.output_dir + '/' + args.log_file, np.array(loglist), newline="\n")

    # Evaluation
    model.eval()
    u = model.base_dist.sample((4 * args.n_sample,))
    samples, _ = model.inverse(u)
    if args.job == "reg":
        samples = samples[samples[:, 1] > 0, :]
        samples = samples[samples[:, 3] > 0, :]
    if samples.size(0) < args.n_sample:
        print("Warning: Insufficient samples")
    else:
        samples = samples[0:args.n_sample, :]

    np.savetxt(args.output_dir + "/generated_data.txt", samples.detach().cpu().numpy())


def execute_VI(args, device, model, optimizer, scheduler):
    from train_VI import train, train_noisy

    if args.job == 'reg':
        from RegressionModels import Regression
        # args.flow_type = 'maf'
        rt = Regression(device)
        rt.NI = 6000
        rt.sigma0 = 0.2
        rt.sigma = 0.2
        data = torch.Tensor(np.loadtxt(args.data_dir))
        rt.X = data[:, 0:4].clone().to(device)
        rt.Y = data[:, 4:9].clone().to(device)
    elif args.job == 'ehr':
        import sys
        sys.path.append('supplMatHarrod20/models/')
        from cvsim6 import cvsim6
        from model_circuit import CircuitModel
        from FNN_surrogate_nested import Surrogate
        from EHR import initialize
        input_map = [4, 8]
        input_size = len(input_map)
        rt = CircuitModel(device, input_size, input_map)
        rt.dbFile = args.data_dir
        raw_data = np.loadtxt(rt.dbFile[:-11] + ".txt")
        initialize(rt, raw_data, input_size)
        rt.surrogate.surrogate_load()
        rt.NI = len(rt.columnID)

    loglist = []

    if args.noisy == 'False':
        for i in range(args.n_iter + 1):
            if scheduler:
                scheduler.step()
            ind = (torch.Tensor(rt.NI).uniform_(0, 1) < args.poisson_ratio).to(device)
            train(model, rt, optimizer, i, args, loglist, ind=ind, sampling=True, surrogate=True)
    else:
        for i in range(args.n_iter + 1):
            if scheduler:
                scheduler.step()
            ind = (torch.Tensor(rt.NI).uniform_(0, 1) < args.poisson_ratio).to(device)
            train_noisy_vi(model, rt, optimizer, device, i, args, loglist, sampling=True, ind=ind, surrogate=True)


if __name__ == "__main__":
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(random.randint(1, 10 ** 9))
    if device.type == 'cuda':
        print("GPU is used.")
        if args.seed >= 0:
            torch.cuda.manual_seed(args.seed)
        else:
            torch.cuda.manual_seed(random.randint(1, 10 ** 9))
    else:
        print("CPU is used.")

    # model
    batch_norm_order = True if args.batch_norm_order == 'True' else False
    if args.flow_type == 'maf':
        model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, None,
                    args.activation_fn, args.input_order, batch_norm=batch_norm_order)
    elif args.flow_type == 'realnvp':  # Under construction
        model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, None,
                        batch_norm=batch_norm_order)
    else:
        raise ValueError('Unrecognized model.')

    model = model.to(device)
    if args.optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.scheduler_order == 'None':
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.task == 'vi':
        execute_VI(args, device, model, optimizer, scheduler)
    else:
        execute_DE(args, device, model, optimizer, scheduler)
