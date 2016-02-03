__author__ = 'Yi Ren'
"""
This contains the class for fitting a sigma for a human player onto EGO.


"""
from ego import Kriging
from preprocess import Preprocess
# from bayes_opt import BayesianOptimization
import numpy as np
import scipy.optimize as opt
import cPickle as pickle
import json
import csv

class CovarianceEstimate:
    '''
    Instantiates meta-problem, i.e., prepares and solves the minimization problem
    to fit a Covariance matrix that maximizes the sum of expected improvement over
    all observed human plays, to fit a Kriging response surface. Currently
    optimizing using SLSQP. Input is reduced 30-dim. (standardized) PCA of original
    18k-length signal

    Parameters:
        X: entire data-set of a Person's plays (n by 30-dim standardized)
        y: array-like of player's scores

    Methods:
        solve(self): actual solver, currently EGO, that finds Sigma[1:30]
            that maximize total expected improvement
    '''
    def __init__(self, X, y):
        self.n = X.shape[1]
        self.model = Kriging(np.ones(self.n)) # Fit Kriging on the data.
        self.model.fit(X, y)
        self.input = X
        self.rem_eng = y
        self.Nfeval = 0

    def callbackF(self, Xi):
        feval = self.model.obj(Xi)
        print '{0:4d} {1: 3.6f}'.format(self.Nfeval, feval)
        if np.any(Xi > 1000.) or np.any(Xi < 0.):
            print 'OoB'
        if feval > 1000.:
            print Xi
        self.Nfeval += 1

    def ego_func(self, x0, x1, x2, x3, x4, x5, x6,
                     x7, x8, x9, x10, x11, x12, x13, x14, x15,
                     x16, x17, x18, x19, x20, x21, x22, x23,
                     x24, x25, x26, x27, x28, x29):
        '''
        Had to take in n individual variables...
        :return: objective function
        '''

        x = np.zeros(self.n)
        args = [x0, x1, x2, x3, x4, x5, x6,
                     x7, x8, x9, x10, x11, x12, x13, x14, x15,
                     x16, x17, x18, x19, x20, x21, x22, x23,
                     x24, x25, x26, x27, x28, x29]
        for n, i in enumerate(args):
            print i
            x[n] = i
        return self.model.obj(x)

    def solve(self):
        '''
        MUST HAVE ego_bds.p and ego_explore.p in directory. This is a list of desired search ranch,
        and the positions to evaluate the model to instantiate EGO. Do not confuse with ego_bounds.txt,
        which is a list of viable ranges for each variable in later simulation (the range of each PC).

        :return: best sigma
        '''

        test_scale = np.arange(0,2,0.5)
        # test_scale = np.array([0.7])
        result_x = np.zeros((test_scale.shape[0], self.n))
        result_f = np.zeros(test_scale.shape)

        for i, s in enumerate(test_scale):

            # x0 = np.zeros(30)+1e-15
            # x0[7] = np.exp(2.5)

            # x0 = np.ones(30)
            # -->
            # x0 = [1.50205418e-01, 6.76120529e-12, 1.84195280e-11, 7.83697468e-12,
            #       1.88757328e-12, 3.95820958e-11, 1.19620582e-11, 1.19440742e-11,
            #       1.87089886e-11, 4.58049765e+00, 4.91223856e-12, 2.84647880e-12,
            #       5.69667522e-12, 2.30425680e-11, 1.14781061e-11, 3.45560046e-11,
            #       9.35167355e-12, 7.56675789e-12, 1.92011109e-03, 1.98174635e-11,
            #       7.22949300e-12, 3.86415689e-12, 3.32313328e-11, 6.81567135e-12,
            #       9.75048750e-12, 1.13677645e-11, 5.17722412e-12, 1.53569991e-03,
            #       2.38163239e-11, 4.80720115e-12] # -6.28698665444

            # x0 = np.ones(30)*0.1
            # -->
            # [  2.00829797e-03   1.90314686e-04   4.95525647e-08   5.72888703e-08
            #    5.05644909e-08   4.96489353e-08   4.86581148e-08   5.14023421e-08
            #    5.30146723e-02   4.28454169e-08   2.13781365e-09   3.81019074e-08
            #    1.58936468e-08   1.02135824e-08   4.68306047e-08   4.19927866e-08
            #    5.23404844e-08   4.40914737e-08   4.98378832e-08   7.80431707e-09
            #    4.60839075e-08   5.09895031e-08   4.21929293e-08   2.49207806e-02
            #    1.62240194e-08   4.79799229e-08   2.22741405e-08   4.69940438e-01
            #    2.09332766e-09   5.85302709e-08] -6.34599415271


            # [ 0.          0.          0.          0.          0.          0.          0.
            #   0.          0.          0.          0.          0.          0.          0.
            #   0.          0.          0.          0.          0.          0.          0.
            #   0.          0.          0.          0.          0.          0.
            #   0.86766298  0.          0.        ] -6.752578221
            x0 = np.ones(self.n)*s
            # x0 = np.random.random(30)*5.

            func = lambda x: - self.model.obj(x)  # use this if switching from EGO

            lb = 0.
            ub = 10.
            bounds = [(lb, ub)]*self.n
            # print bounds

            # these are some alternative functions, which use 'callbackF for verbosity'
            # print self.model.obj(x0)
            print 'Initializing at '+str(s)
            res = opt.minimize(func, x0=x0, bounds=bounds, method='SLSQP', tol=1e-5,
                               options={'eps': 5e-4, 'iprint': 2, 'disp': True, 'maxiter': 100})
            # res = opt.differential_evolution(func, bounds, disp=True, popsize=10)
            # res = opt.basinhopping(func, x0=x0, disp=True)

            # bounds = pickle.load(open("ego_bds.p", "rb"))  # load sigma boundaries
            # bo = BayesianOptimization(self.ego_func, pbounds=bounds)  # create optimizer object
            # explore = pickle.load(open("ego_explore.p", "r"))  # load points to check
            # bo.explore(explore) #initiate
            # bo.maximize(init_points=15, n_iter=25)
            # print bo.res['max']
            print res.x, res.fun
            # return bo.res['max']
            result_f[i] = res.fun
            result_x[i] = res.x
        return result_f, result_x

# #
# # get data from the game
# # delete the parameters if performing first-time or new player.
# # Parameters are there to speed up after saving a pkl.
pre = Preprocess(pca_model='../eco_full_pca.pkl', all_dat='../all_games.pkl')
# pre = Preprocess()
# pre.get_json('alluser_control.json')  # uncomment this to create the pkl file needed!!
# pre.train_pca()
X, y = pre.ready_player_one(3)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)
#
# # get sigma estimate that maximizes the sum of expected improvements
soln = CovarianceEstimate(X, y)
print soln.model.obj([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      7.406084929986003e-08, 0.0, 0.0, 5.084319587668944e-08, 0.0, 0.0, 0.0, 0.0,
                      3.318550043334831e-07, 0.0, 0.0, 0.0, 3.644883020460128])
[obj_set, sigma_set] = soln.solve()

# # pick the best solution
obj = obj_set.min(axis=0)
sigma = sigma_set[obj_set.argmin(axis=0), :]
print obj, sigma

# # load bounds
# from numpy import loadtxt
# bounds = loadtxt("ego_bounds.txt", comments="#", delimiter=",", unpack=False)
#
# # store sigma for simulation
# # TODO: need to specify file name based on settings, e.g., optimization algorithm and input data source (best player?)

file_address = 'p3_slsqp_sigma.json'
with open(file_address, 'w') as f:
    # pickle.dump([obj_set, sigma_set], f)
    json.dump([obj, sigma.tolist()], f, sort_keys=True, indent=4, ensure_ascii=False)
f.close()


with open('p3_range_transform.json', 'w') as outfile:
    json.dump({'range':scale.scale_.tolist(), 'min':scale.min_.tolist()},
              outfile, sort_keys = True, indent = 4, ensure_ascii=False)
with open('p3_ICA_transform.json', 'w') as outfile:
    json.dump({'mix':pre.pca.mixing_.tolist(), 'unmix':pre.pca.components_.tolist(), 'mean':pre.pca.mean_.tolist()},
              outfile, sort_keys = True, indent = 4, ensure_ascii=False)
# A = pre.pca.components_
# Std_inv = np.diag(1/scale.std_)
# vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
# np.savetxt('visualize_this.txt', vis)