__author__ = 'p2admin'
"""
This contains the class for fitting a sigma for a human player onto EGO.
"""
from ego import EfficientGlobalOpt
#TODO: include preprocessing code for the raw game data
# from preprocess import Preprocess
import numpy as np
import scipy.optimize as opt
import pickle

class CovarianceEstimate:
    def __init__(self, X, y):
        self.model = EfficientGlobalOpt()
        self.model.X = X
        self.model.y = y
        self.model.n, self.model.p = X.shape

    def solve(self):
        x0 = np.ones(n)
        func = self.model.f
        lb = 0
        ub = np.inf
        bounds = [(lb, ub)]*self.model.p
        result = opt.fmin_slsqp(func=func, x0=x0, bounds=bounds)
        return result.out

# get data from the game
X, y = Preprocess()

# get sigma estimate that maximizes the sum of expected improvements
sigma = CovarianceEstimate(X, y).solve()

# store sigma for simulation
# TODO: need to specify file name based on settings, e.g., optimization algorithm and input data source (best player?)
file_address = 'sigma.pickle'
with open(file_address, 'w') as f:
        pickle.dump(sigma, f)
f.close()







