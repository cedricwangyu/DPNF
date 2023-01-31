import os
import torch
import numpy as np

from model_circuit import CircuitModel
from FNN_surrogate_nested import Surrogate
from utility import parse

args = parse('EHR')
torch.set_default_tensor_type(torch.DoubleTensor)


def initialize(rg, raw_data, input_size):
    rg.columnID = list(np.where(raw_data[:, -3] * 2 / 3 + raw_data[:, -2] / 3 > 20)[0])
    if len(rg.columnID) > 43:
        rg.columnID = rg.columnID[:43]

    rg.rowID = [i for i in range(16) if i not in (0, 5, 6, 7, 8, 9, 10, 11)]
    rg.outputID = [i for i in range(24) if i not in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]
    rg.data = \
        np.loadtxt(rg.dbFile, delimiter=',', usecols=(col_id + 1 for col_id in rg.columnID)).reshape(16, -1)[
            rg.rowID,]
    rg.stds[rg.outputID] = rg.data.std(1)
    rg.surrogate = Surrogate(model_name="circuit_model",
                             model_func=lambda x: rg.solve_t(rg.transform(x)),
                             input_size=input_size,
                             output_size=len(rg.rowID),
                             limits=torch.Tensor([[-7, 7]] * input_size),
                             memory_len=20)


def EHR_VI_preprocess(args):
    device = torch.device('cpu')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    input_map = [4, 8]
    input_size = len(input_map)
    rg = CircuitModel(device, input_size, input_map)

    raw_data = np.loadtxt(args.data_dir)
    data = np.transpose(raw_data)
    idx = [i for i in range(19) if i not in (10, 11, 12)]
    data = data[idx]
    data = data[:, np.logical_and(data[-4, :] <= 100, data[-4, :] >= 0)]
    data = data[:, np.logical_and(data[-3, :] <= 80, data[-3, :] >= 0)]
    data = data[:, np.logical_and(data[-2, :] <= 80, data[-2, :] >= 0)]
    data = data[:, np.logical_and(data[-1, :] <= 80, data[-1, :] >= 0)]
    feature_names = rg.circuit_model.resName[[0, 1, 2, 3, 4, 5, 7, 8, 11, 13, 14, 15, 20, 21, 22, 23]]
    output_file = np.concatenate((np.array(feature_names).reshape(-1, 1), data), axis=1)
    np.savetxt(args.data_dir[:-4] + "_cvsim6.txt", output_file, delimiter=',', fmt='%s')

    rg.dbFile = args.data_dir[:-4] + "_cvsim6.txt"
    initialize(rg, raw_data, input_size)

    grid_limits = torch.Tensor([[-7, 7]] * input_size)
    gridnum = 100
    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    gridlist = soboleng.draw(gridnum)
    gridlist[:, 0] = gridlist[:, 0] * (grid_limits[0, 1] - grid_limits[0, 0]) + grid_limits[0, 0]
    gridlist[:, 1] = gridlist[:, 1] * (grid_limits[1, 1] - grid_limits[1, 0]) + grid_limits[1, 0]
    rg.surrogate.pre_grid = gridlist
    rg.surrogate.pre_out = rg.surrogate.mf(rg.surrogate.pre_grid)
    rg.surrogate.pre_train(120000, 0.01, 0.9999, 500, store=True, reg=True)
    rg.surrogate.surrogate_save()
    print(rg.surrogate.pre_grid)
    print(rg.surrogate.pre_out)
    rg.surrogate.surrogate_load()


if __name__ == "__main__":
    EHR_VI_preprocess(args)
