__author__ = 'Yi Ren'
"""
This contains the class for fitting a sigma for a human player onto EGO.


"""
from ego import Kriging
from preprocess import Preprocess
from bayes_opt import BayesianOptimization
import numpy as np
import scipy.optimize as opt
import cPickle as pickle

class CovarianceEstimate:
    '''
    Instantiates meta-problem, i.e., prepares and solves the minimization problem
    to fit a Covariance matrix that maximizes the sum of expected improvement over
    all observed human plays, to fit a Kriging response surface. Currently
    optimizing using EGO. Input is reduced 30-dim. (standardized) PCA of original
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

        # x0 = 1./np.ones(30)/9.2564178897127807

        # func = lambda x: - self.model.obj(x)  # use this if switching from EGO

        # lb = 0.
        # ub = 1000.
        # bounds = [(lb, ub)]*self.n
        # print bounds

        # these are some alternative functions, which use 'callbackF for verbosity'
        # print self.model.obj(x0)
        # res = opt.minimize(func, x0=x0, bounds=bounds, method='SLSQP', callback=self.callbackF, tol=1e-8)
        # res = opt.differential_evolution(func, bounds, disp=True, popsize=10)
        # res = opt.basinhopping(func, x0=x0, disp=True)

        bounds = pickle.load(open("ego_bds.p", "rb"))  # load sigma boundaries
        bo = BayesianOptimization(self.ego_func, pbounds=bounds)  # create optimizer object
        explore = pickle.load(open("ego_explore.p", "rb"))  # load points to check
        bo.explore(explore) #initiate
        bo.maximize(init_points=15, n_iter=25)
        print bo.res['max']
        # print res.x, res.fun
        return bo.res['max']

#
# get data from the game
# delete the parameters if performing first-time or new player.
# Parameters are there to speed up after saving a pkl.
pre = Preprocess(pca_model='eco_full_pca.pkl', all_dat='all_games.pkl')
# pre.get_json('alluser_control.json')  # uncomment this to create the pkl file needed!!

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
