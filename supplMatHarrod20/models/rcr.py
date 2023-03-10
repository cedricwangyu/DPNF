import numpy as np
from circuitModels import circuitModel,mmHgToBarye,baryeTommHg
from c_routines import evalDeriv_rcr
import matplotlib.pyplot as plt

class rcrModel(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None,
               debugMode=False,verbose=False):    
    # Init parameters
    numParam    = 3
    numState    = 1
    numAuxState = 5
    numOutputs  = 3
    parName = ["R1","R2","C"]
    icName  = np.array(["p_ini"])
    resName = np.array(["min_pressure",
                        "max_pressure",
                        "avg_pressure"])
    limits = np.array([[100.0, 1500.0],
                       [100.0, 1500.0],
                       [1.0e-5, 1.0e-2]])
    defIC = np.array([55.0*mmHgToBarye])
    defParam = np.array([1000.0,1000.0,0.00005])
    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     icName,parName,resName,
                     limits,defIC,defParam,
                     cycleTime,totalCycles,forcing=forcing,
                     debugMode=debugMode,verbose=verbose)

  def evalDeriv(self,t,y,params):
    return evalDeriv_rcr(t,y,params,
                        self.cycleTime,self.forcing,
                        self.numState,self.numAuxState)

  def plot_model(self,t,y,aux,start,stop):
    # Plot results
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    plt.plot(t,aux[2]/mmHgToBarye,label='p_0')
    plt.plot(t,y[0]/mmHgToBarye,label='p_1')
    plt.plot(t,aux[1]/mmHgToBarye,label='p_d')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [mmHg]')
    plt.subplot(1,2,2)
    plt.plot(t,aux[3]/mmHgToBarye,label='Q1')
    plt.plot(t,aux[4]/mmHgToBarye,label='Q2')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Flow [mL/s]')
    plt.tight_layout()
    plt.show()
    plt.close()

  def postProcess(self,t,y,aux,start,stop):    
    res = np.zeros(self.numOutputs)
    # Compute Min Pressure
    res[0] = np.min(aux[2,start:stop])/mmHgToBarye
    # Compute Max Pressure
    res[1] = np.max(aux[2,start:stop])/mmHgToBarye
    # Compute Average Pressure
    res[2] = (np.trapz(aux[2,start:stop],t[start:stop])/float(self.cycleTime))/mmHgToBarye

    return res

# TESTING MODEL
if __name__ == "__main__":

  # Create the model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('../data/inlet.flow')
  model = rcrModel(cycleTime,totalCycles,forcing,debugMode=True)

  # Get Default Initial Conditions
  y0        = model.defIC
  # Get Default Parameters
  params    = model.defParam
  # Solve Model and Get Results
  outs      = model.solve(params=params,y0=y0)[0]
  outLabels = model.resName

  # Array with measurement standard deviations - same size of the model result vector
  stds = np.array([1.0,  # min_pressure
                   1.0,  # max_pressure
                   1.0]) # avg_pressure

  # Evaluate Model Log-Likelihood
  # dbFile = '../data/EHR_dataset.csv'
  dbFile = '../data/rc_dataset.csv'
  columnID = 0 # First Patient

  ll = model.evalNegLL(columnID,dbFile,stds,params,y0)

  print("Model Negative LL: ",ll)

