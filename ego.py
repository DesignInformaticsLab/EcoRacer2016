__author__ = 'p2admin'
import numpy as np
from scipy.stats import norm


class EfficientGlobalOpt(self):

"""
This is the actual optimization class, that will interface with the higher level regression.
"""
    def __init__(self, sig):
        self.Sigma = sig
        self.X = np.array([])

    def step(self, Xi, yi):
    #add a point to the model

    def fit(self, X, y):
    #step on all existing points. in self.X
        for i, n in enumerate(X):


    def kernel(v1, v2, sig):
        # helper function for step. sig assumed to be vector of diagonals
        Sigma = np.diag(sig) #must be invertible.
        S_inv = np.linalg.inv(Sigma)

        arg = -1*(np.subtract(x1, x2).T).dot(S_inv.dot(np.subtract(x1, x2)))
        return np.exp(arg)

    def z_score(self, fmin, y, s):
        return (fmin - y)/s

    def expected_improv(fmin, y, s):
        z = z_score(fmin, y, s)
        pdf = norm.pdf(z)
        cdf = norm.cdf(z)

        return np.multiply(np.subtract(fmin, y), pdf) + np.multiply(s, cdf)

    def mse(self, v1, v2, f):
    


