import sys, os

sys.path.append('supplMatHarrod20/models/')
from cvsim6 import cvsim6
import numpy as np
import torch
from FNN_surrogate_nested import Surrogate

torch.set_default_tensor_type(torch.DoubleTensor)


# This is the parent class for all models
class model_generic(object):
    # Init model
    def __init__(self, device):
        self.device = device

    # Left parameter unchanged by defualt
    def map_params(self, par, mu, bounds):
        return par

    # If not implemented by the model, do not plot
    def plot_results(self, exp):
        pass


class CircuitModel(model_generic):
    def __init__(self, device, input_size, input_map):

        #  Invoke Superclass Constructor
        super().__init__(device)

        # Set data members
        self.cycleTime = 1.07
        self.totalCycles = 10
        self.circuit_model = cvsim6(self.cycleTime, self.totalCycles, debugMode=False)

        if (input_map is None):
            self.input_map = [i for i in range(input_size)]
            if not (input_size == self.circuit_model.defParam.shape[0]):
                print('ERROR: When input_map is None, the number of parameters must agree.')
        else:
            # Plot list of used parameters
            print('Using a subset of the input parameters.')
            print('%-5s %-25s' % ('IDX', 'Parameter'))
            for loopA in input_map:
                print('%-5d %-25s' % (loopA, self.circuit_model.parName[loopA]))
            self.input_map = input_map

        # Measurement standard deviations
        self.stds = np.array([3.0,  # ip_0002_heart_rate2
                              1.5,  # ip_0002_systolic_bp_2
                              1.5,  # ip_0002_diastolic_bp_2
                              0.2,  # ip_0002_cardiac_output
                              50.0,  # ip_0002_systemic_vascular_resistan
                              5.0,  # ip_0002_pulmonary_vascular_resista
                              0.5,  # ip_0002_cvp
                              1.0,  # ip_0002_right_ventricle_diastole
                              1.0,  # ip_0002_right_ventricle_systole
                              1.0,  # left_ventricle_diastole
                              1.0,  # left_ventricle_systole
                              1.0,  # ip_0002_rvedp
                              0.5,  # ip_0002_aov_mean_pg
                              0.5,  # ip_0002_aov_peak_pg
                              # 6.0,  # ip_0002_mv_decel_time
                              # 0.2,  # ip_0002_mv_e_a_ratio
                              # 6.0,  # ip_0002_pv_at
                              0.5,  # ip_0002_pv_max_pg
                              0.5,  # ip_0002_ra_pressure
                              3.0,  # ip_0002_ra_vol_a4c - End Systolic
                              3.0,  # ip_0002_la_vol_a4c - End Systolic
                              10.0,  # ip_0002_lv_esv
                              20.0,  # ip_0002_lv_vol
                              2.0,  # ip_0002_lvef
                              1.0,  # ip_0002_pap_diastolic
                              1.0,  # ip_0002_pap_systolic
                              1.0])  # ip_0002_wedge_pressure

        # HEALTHY VALIDATION PATIENT
        self.dbFile = 'source/data/circuit_model_output.txt'
        self.columnID = [0]
        self.rowID = [i for i in range(24)]
        self.outputID = [i for i in range(24)]
        self.data = \
            np.loadtxt(self.dbFile, delimiter=',', usecols=(col_id + 1 for col_id in self.columnID)).reshape(
                len(self.rowID), -1)[
                self.rowID,]
        self.surrogate = Surrogate(model_name="circuit_model",
                                   model_func=lambda x: self.solve_t(self.transform(x)),
                                   input_size=input_size,
                                   output_size=len(self.rowID),
                                   limits=torch.Tensor([[-7, 7]] * input_size),
                                   memory_len=20)
        self.NI = len(self.columnID)

    def transform(self, x):
        return torch.tanh(x / 7 * 3) * 50

    def inverse_transoform(self, y):
        return torch.atanh(y / 50) * 7 / 3

    def solve_t(self, params, print_mode=True):
        # params should be genuine parameters of the model.
        # If it is not, self.transform should be called before self.solve_t
        full_params = np.zeros((len(params), self.circuit_model.numParam))
        full_params[:, self.input_map] = params.detach().cpu().numpy()
        mapped_par = self.map_params(full_params)
        outs = []
        if print_mode:
            for loopA in range(len(params)):
                output, _, _ = self.circuit_model.solve(mapped_par[loopA], y0=None)
                outs.append(output)
        else:
            for loopA in range(len(params)):
                output, _, _ = self.circuit_model.solve(mapped_par[loopA], y0=None)
                outs.append(output)
        return torch.Tensor(outs)[:, self.outputID]

    def den_t(self, params, surrogate=True, ind=None):
        if surrogate:
            out = self.surrogate.forward(params)
        else:
            out = self.solve_t(self.transform(params))
        Data = torch.Tensor(self.data) if ind is None else torch.Tensor(self.data[:, ind])
        stds = torch.Tensor(self.stds)[self.outputID]
        # Eval LL
        ll1 = -0.5 * Data.size(0) * Data.size(1) * np.log(2.0 * np.pi)  # a number
        ll2 = (-0.5 * Data.size(1) * torch.log(torch.prod(stds))).item()  # a number
        ll3 = - 0.5 * torch.sum(torch.sum((out.unsqueeze(0) - Data.t().unsqueeze(1)) ** 2, dim=0) / stds ** 2,
                                dim=1, keepdim=True)
        negLL = -(ll1 + ll2 + ll3)
        adjust = torch.log(1 - torch.tanh(params / 7 * 3) ** 2).sum(1, keepdim=True)
        return - negLL + adjust

    def map_params(self, par):
        # Get default model parameters and bounds
        mu = self.circuit_model.defParam
        bounds = self.circuit_model.limits
        # Map only the parameters that are not frozen
        res = np.zeros(par.shape)
        for loopA in range(len(par)):
            for loopB in range(len(par[loopA])):
                # if(par[loopA,loopB] > 0):
                #  res[loopA,loopB] = mu[loopB] + par[loopA,loopB]*(bounds[loopB,1] - mu[loopB])/10.0
                # else:
                #   res[loopA,loopB] = mu[loopB] + par[loopA,loopB]*(mu[loopB] - bounds[loopB,0])/10.0
                res[loopA, loopB] = mu[loopB] + par[loopA, loopB] * (bounds[loopB, 1] - bounds[loopB, 0]) / 50.0
                # res[loopA, loopB] = par[loopA, loopB] * (bounds[loopB, 1] - bounds[loopB, 0])
        return res

    def gen_measurements(self):
        # Get true parameter set
        true_params = self.circuit_model.defParam
        mean_outputs, _, _ = self.circuit_model.solve(true_params, y0=None)
        samples = np.random.multivariate_normal(mean_outputs, np.diag(self.stds), size=50)
        output_names = self.circuit_model.resName
        output_file = np.concatenate((np.array(output_names).reshape(-1, 1), np.transpose(samples)), axis=1)
        # print(output_file)
        np.savetxt('source/data/circuit_model_output.txt', output_file, delimiter=',', fmt='%s')





if __name__ == "__main__":
    # test_model()
    # EHR()
    bulk_run()
