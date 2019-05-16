import pyLLE
import numpy as np

res = {'R': 23e-6, # ring radius in meter
       'Qi': 1e6,  # Intrinsic Q factor
       'Qc': 1e6,  # Coupled Q factor
       'γ': 1.55,  # Non-linear coefficient at the pump frequency
       'dispfile': 'TestDispersion.txt', # frequency and corresponding azymuthal mode simulated previously
    }


sim = {'Pin': 150e-3, # Input power in Q
       'Tscan': 1e6,  # Length of the simulation in unit of round trip
       'f_pmp': 191e12, # Pump Frequency
       'δω_init': 2e9*2*np.pi, # Initial detuning of the pump in rad/s
       'δω_end': -8e9*2*np.pi,  # End detunin of the pump in rad/s
       'μ_sim': [-74,170],  # azimuthal mode to simulate on the left and right side of the pump
       'μ_fit': [-71, 180], # azimuthal mode to fit the dispersion on the left and right side of the pump
    }

solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
solver.Analyze(plot=True, plottype='all')
# solver.Setup()
# solver.SolveTemporal()
# solver.RetrieveData()
# solver.PlotCombPower()