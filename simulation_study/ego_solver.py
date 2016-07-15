__author__ = 'Thurston Sexton'
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import pinv2, inv
import scipy.optimize as opt
from pyDOE import lhs

class EGO():

    """
    This is the actual optimization class, that will interface with the higher level regression.
    """
    def __init__(self, sig_inv, obj, bounds, max_iter, num_ini_guess):
        self.SI = np.diag(sig_inv)
        self.X = np.array([[]]) # observed inputs, column vars
        self.y = np.array([[]]) # observed scores (must be 2-D)
        self.R = np.array([[]])
        self.RI = np.array([[]])
        self.b = np.array([])
        self.obj = obj# input optimization problem
        self.bounds = bounds # bounds of the problem
        self.p = bounds.shape[0] # problem size
        self.max_iter = max_iter # EGO max iter
        self.num_ini_guess = num_ini_guess # initial sample size
        self.initial_sample() # samples to start EGO

    def solve(self):
        while not self.terminate():
            new_x = self.sample()
            if new_x.shape[1] < 2:
                break
            new_y = self.obj(*new_x[0, :])
            print new_y
            self.X = np.vstack((self.X, new_x))
            self.y = np.hstack((self.y, new_y))
        return [self.X, self.y]

    def terminate(self):
        if self.X.shape[0] < self.max_iter:
            print "NOW FINDING SAMPLE ", self.X.shape[0]
            return False
        else:
            return True

    def initial_sample(self):
        self.X = lhs(self.p, self.num_ini_guess)
        self.X = self.X*(self.bounds[:, 1]-self.bounds[:, 0])+self.bounds[:, 0]
        # self.y = self.obj(self.X[:, 0], self.X[:, 1])
        self.y = self.obj(*self.X.T)

    def fit(self):
        self.R = self.R_ij(self.X)
        if np.linalg.matrix_rank(self.R) < self.R.shape[1]:
            wait = 1.
        try:
            self.RI = pinv2(self.R)
        except:
            wait = 1.

        self.b = self.get_b()

    def R_ij(self, X):
        # kernel for non-identity cov. matrix (sigma)
        dists = squareform(pdist(X, 'mahalanobis', VI=self.SI))
        self.R = np.exp(-1*dists)
        return self.R

    def r_i(self, x, X):
        # kernel for non-identity cov. matrix (sigma)
        # X, x must be 2-D!
        if x.size == X.shape[1]:#ensure is 2D
            x = np.array(x, ndmin=2)

        dists = cdist(X, x, 'mahalanobis', VI = self.SI)
        return np.exp(-1*dists)

    def get_b(self):
        dim = np.size(self.y)
        ones = np.ones(dim)
        num = ones.T.dot(self.RI.dot(self.y))
        den = ones.T.dot(self.RI.dot(ones))
        return num/den

    def yhat(self, x):
        # the kriging surface for given X,y
        r = self.r_i(x, self.X)
        return self.b + r.T.dot(self.RI.dot(np.subtract(self.y, self.b)))

    def get_s(self, x): # Uncertainty function
        dim = np.size(self.y)
        ones = np.ones(dim)

        r = self.r_i(x, self.X)
        if np.linalg.matrix_rank(self.R) < self.R.shape[1]:
            return 0

        # WARNING: this is getting runtime warnings (invalid value encountered in divide)
        mse = np.max((0,
                      1-r.T.dot(self.RI.dot(r))+(1.-ones.T.dot(self.RI.dot(r)))**2/(ones.T.dot(self.RI.dot(ones)))))

        sig = np.sqrt((self.y-self.b).T.dot(self.RI.dot(self.y-self.b))/dim)

        if mse==0 or sig==0:
            wait = 1

        try:
            np.sqrt(mse)
        except:
            wait = 1

        # WARNING: this is getting runtime warnings (invalid value encountered in multiply/sqrt)
        return np.sqrt(mse)*sig

    def concentrated_likelihood(self, proposed_SIs):
        ls = np.zeros(proposed_SIs.shape[0])  # the likelihood array
        # old_sig = self.SI.copy()

        for i, prop in enumerate(proposed_SIs):
            self.SI = np.diag(prop)
            self.fit()

            dim = np.size(self.y)
            sig = np.sqrt((self.y - self.b).T.dot(self.RI.dot(self.y - self.b)) / dim)

            den = (2.*np.pi)**(dim/2.)*(sig**2)**(dim/2.)*np.linalg.det(self.R)**0.5
            ls[i] = np.log(den)+dim/2.
            # ls[i] = np.exp(-dim/2.)/den  # eq. 4 Jones '98

        # self.SI = np.diag(proposed_SIs[np.argmax(ls)])  # lambda with highest likelihood
        self.SI = np.diag(proposed_SIs[np.argmin(ls)])  # lambda with highest likelihood

    def f(self, x): # Expected improvement function
        if x.size == self.X.shape[1]:
            x = np.array(x, ndmin=2) #ensure is 2D
        # expected improvement function, given the model y_h
        y_h = self.yhat(x)
        ymin = np.min(self.y)

        s = self.get_s(x)
        z = np.divide(np.subtract(ymin, y_h), s+1e-12)
        if s==0:
            pdf = 0
        else:
            pdf = norm.pdf(z)
        cdf = norm.cdf(z)
        f_x = np.multiply(np.subtract(ymin, y_h), cdf) + np.multiply(s, pdf)
        if abs(f_x)<1e-3:
            wait = 1

        return f_x

    def sample(self): # sample with max expected improvement
        self.opt_ini_guess = lhs(self.p, 10) # samples for internal max expectation
        self.opt_ini_guess = self.opt_ini_guess*(self.bounds[:, 1]-self.bounds[:, 0])+self.bounds[:, 0]
        p = self.opt_ini_guess.shape[1] # problem size
        n = self.opt_ini_guess.shape[0] # number of optimization runs
        test_lambdas = np.array([int(self.p)*[0.01],
                                 int(self.p)*[0.1],
                                 int(self.p)*[1.0],
                                 int(self.p)*[10.0]])
        self.concentrated_likelihood(test_lambdas)
        self.fit()
        func = lambda x: - self.f(x)
        result_x = np.zeros((n, p))
        result_f = np.zeros((n, 1))
        for i, x0 in enumerate(self.opt_ini_guess):
            res = opt.minimize(func, x0=x0, bounds=self.bounds, method='slsqp', tol=1e-5,
                                   options={'eps': 1e-8, 'iprint': 2, 'disp': False, 'maxiter': 100})
            result_f[i] = res.fun
            result_x[i] = res.x
            if np.any(np.isnan(res.x)==True):
                wait = 1

        if np.abs(np.min(result_f, axis=0)) < 1e-3:
            return np.zeros((1,1)) # terminate if no improvement

        return result_x[np.argmin(result_f, axis=0)]

