__author__ = 'Yi Ren'
"""
This contains the class for fitting a sigma for a human player onto EGO.
"""
from ego import Kriging
from preprocess import  Preprocess


import numpy as np
import scipy.optimize as opt
import pickle

class CovarianceEstimate:
    def __init__(self, X, y):
        self.n = X.shape[1]
        self.model = Kriging(np.ones(self.n))
        self.model.fit(X,y)
        self.input = X
        self.rem_eng = y

    def solve(self):
        x0 = 1./np.ones(30)/18.927661099834339

        func = lambda x: - self.model.obj(x)
        lb = 0
        ub = 100
        bounds = [(lb, ub)]*self.n
        # print bounds
        print self.model.obj(x0)
        res = opt.minimize(func, x0=x0, bounds=bounds, method='SLSQP')
        #res = opt.differential_evolution(func, bounds, disp=True, popsize=50)
        # res = opt.basinhopping(func, x0=x0, disp=True)
        print res.x, res.fun
        return res.x

# get data from the game
pre = Preprocess(pca_model='eco_full_pca.pkl', all_dat='all_games.pkl')
# pre.get_json('alluser_control.json')  # uncomment this to get the pkl file needed!!

X, y = pre.ready_player_one(2)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X = scale.fit_transform(X)

# get sigma estimate that maximizes the sum of expected improvements
sigma = CovarianceEstimate(X, y).solve()

# store sigma for simulation
# TODO: need to specify file name based on settings, e.g., optimization algorithm and input data source (best player?)
file_address = 'sigma.pickle'
with open(file_address, 'w') as f:
        pickle.dump(sigma, f)
f.close()

A = pre.pca.components_
Std_inv = np.diag(1/scale.std_)
vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
np.savetxt('visualize_this.txt', vis)
