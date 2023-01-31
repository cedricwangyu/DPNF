import os
import numpy as np
import torch
import torch.distributions as D

torch.set_default_tensor_type(torch.DoubleTensor)


class Regression:
    def __init__(self, device):
        self.device = device
        self.sigma0 = 0.2
        self.sigma = 0.2
        self.NI = 6000
        self.NJ = 5
        self.defbeta = torch.Tensor([0.2, 1.0, 0.8, -1.2, 0.6]).to(device)
        self.data_dir = "source/data/data_reg.txt"
        self.X = None
        self.Y = None
        if os.path.exists(self.data_dir):
            data = torch.Tensor(np.loadtxt(self.data_dir))
            self.X = data[:, 0:4].clone().to(device)
            self.Y = data[:, 4:9].clone().to(device)
        self.defout = self.solve(self.defbeta.unsqueeze(0)) if self.X is not None else None
        self.Cov = (torch.Tensor(self.NJ, self.NJ).fill_(self.sigma0 ** 2) + torch.eye(self.NJ) * self.sigma ** 2).to(
            device)
        self.dist = D.MultivariateNormal(loc=torch.zeros(self.NJ).to(device), covariance_matrix=self.Cov)
    def gen_data(self):
        if os.path.exists(self.data_dir):
            print("Data already exists.")
            return
        self.X = torch.Tensor(self.NI, 4).uniform_(0, 1) * torch.Tensor([1.0, 3.0, 0.5, 2.0])
        self.Y = self.defout.reshape(-1, 1) \
              + torch.normal(0, self.sigma0, size=(self.NI, 1)) \
              + torch.normal(0, self.sigma, size=(self.NI, self.NJ))

        np.savetxt(self.data_dir, torch.cat([self.X, self.Y], dim=1).detach().cpu().numpy())

    def solve(self, param, ind=None):
        X = self.X if ind is None else self.X[ind, :]
        b0, b1, b2, b3, b4 = torch.split(param, 1, dim=1)
        x1, x2, x3, x4 = torch.split(X.t(), 1, dim=0)
        # print(x1.size(), b1.size())
        Ey = b0 \
             + torch.exp(x1 * b1) \
             + torch.log(x2 + torch.exp(b2)) \
             + b3 * torch.exp(x3) \
             + torch.log(x4 + torch.exp(b4))
        return Ey

    def den_t(self, param, ind=None, surrogate=False):
        if self.X is None or self.Y is None: raise ValueError('Absence of X and Y data.')
        if ind is None:
            diff = (self.solve(param, ind=ind).unsqueeze(0) - self.Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, self.Y.size(0)).sum(1).reshape(-1, 1)
        else:
            Y = self.Y[ind, :]
            diff = (self.solve(param, ind=ind).unsqueeze(0) - Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, Y.size(0))
        return res
