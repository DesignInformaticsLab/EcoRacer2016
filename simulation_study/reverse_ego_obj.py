__author__ = 'Thurston Sexton'
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import pinv2, inv
from pyDOE import lhs
from scipy.misc import logsumexp

class Kriging():

    """
    This is the actual optimization class, that will interface with the higher level regression.
    """
    #def __init__(self, sig, X, y):
    def __init__(self, sig_inv, bounds, num_ini_guess):
        #self.X = X
        #self.y = y

        self.SI = np.diag(sig_inv)
        #self.Sigma = sig
        self.X = np.array([[]]) # observed inputs, column vars
        self.y = np.array([[]]) # observed scores (must be 2-D)
        self.recent_path = np.array([])

        # self.model = np.array([])

        self.bounds = bounds
        self.num_ini_guess = num_ini_guess
        # setup random samples to calculate mean of expected improvement
        # self.samples = lhs(2, 100)  # for 2-dim funcs
        self.samples = lhs(6, 100)  # for 6-dim rosenbrock
        self.samples = self.samples*(self.bounds[:, 1]-self.bounds[:, 0])+self.bounds[:, 0]


    #def fit(self):
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.R = self.R_ij(self.X)
        #print self.R
        if np.linalg.matrix_rank(self.R) < self.R.shape[1]:
            wait = 1.
        self.RI = pinv2(self.R)
        # self.RI = inv(self.R)
        #print self.RI
        self.b = self.get_b()

    def R_ij(self, X):
        # kernel for non-identity cov. matrix (sigma)
        dists = squareform(pdist(X, 'mahalanobis', VI=self.SI))
        self.R = np.exp(-1*dists)
        return self.R

    def r_i(self, x, X):
        # kernel for non-identity cov. matrix (sigma)
        # X, x must be 2-D!
        if x.size==X.shape[1]:#ensure is 2D
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

    def get_s(self, x):
        dim = np.size(self.y)
        ones = np.ones(dim)

        r = self.r_i(x, self.X)
        if np.linalg.matrix_rank(self.R) < self.R.shape[1]:
            return 0
        # WARNING: this is getting runtime warnings (invalid value encountered in divide)
        mse = np.max((0,
                      1-r.T.dot(self.RI.dot(r))+(1.-ones.T.dot(self.RI.dot(r)))**2/(ones.T.dot(self.RI.dot(ones)))))

        sig = np.sqrt((self.y-self.b).T.dot(self.RI.dot(self.y-self.b))/dim)

        if mse<0:
            wait = 1

        try:
            np.sqrt(mse)
        except:
            wait = 1

        # WARNING: this is getting runtime warnings (invalid value encountered in multiply/sqrt)
        return np.sqrt(mse)*sig

    def f(self, x):
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
        return f_x

    def f_path(self, sig_inv):
        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]
        old_sig = self.SI[:]

        #self.Sigma = sig  # replace stored sigma with supplied
        self.SI = np.diag(sig_inv)
        path = np.zeros(self.n)
        # print self.n, path.shape
        self.fit(old_X[:self.num_ini_guess],old_y[:self.num_ini_guess])  # first observation
        for i, x in enumerate(old_X[self.num_ini_guess:], self.num_ini_guess):

            path[i-1] = self.f(x)
            self.fit(old_X[:i+1], old_y[:i+1])

        # return original database to storage
        self.X = old_X
        self.y = old_y
        self.SI = old_sig

        # return np.nan_to_num(path)
        return path

    def sampled_f_path(self, sig_inv, samples):
        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]
        old_sig = self.SI[:]

        # self.Sigma = sig  # replace stored sigma with supplied
        self.SI = np.diag(sig_inv)

        sample_size = samples.shape[0]
        sampled_path = np.zeros((self.n, sample_size))
        # print self.n, path.shape
        self.fit(old_X[:self.num_ini_guess],old_y[:self.num_ini_guess])  # first observation
        for i, x in enumerate(old_X[self.num_ini_guess:], self.num_ini_guess):
            for j, xx in enumerate(samples):
                sampled_path[i-1,j] = self.f(xx)
            self.fit(old_X[:i+1], old_y[:i+1])

        # return original database to storage
        self.X = old_X
        self.y = old_y
        self.SI = old_sig
        return sampled_path

    def obj(self, sig_inv, alpha):
        path = self.f_path(sig_inv)
        sampled_path = self.sampled_f_path(sig_inv, self.samples)
        # log_prob = np.log(1./(1.+np.sum(np.exp(alpha*(sampled_path.T - path)), axis=0)))
        # sum_improv = np.sum(self.recent_path)

        log_prob = alpha*path - logsumexp(alpha*np.vstack((sampled_path.T, path)).T, axis=1)
        self.recent_path = log_prob
        sum_improv = np.sum(log_prob)

        return sum_improv

        # # save whole database as a copy
        # old_X = self.X[:]
        # old_y = self.y[:]
        # old_sig = self.SI[:]
        #
        # #self.Sigma = sig  # replace stored sigma with supplied
        # self.SI = np.diag(sig_inv)
        # sum = 0.  # initiate loop
        # self.fit(old_X[:1],old_y[:1])  # first observation
        # for i, x in enumerate(old_X, 1): #loop through the rest
        #     f_current = self.f(x)  # expected improvement for current obs.
        #     self.fit(old_X[:i], old_y[:i])  # fit up to current obs.
        #     sum += np.nan_to_num(f_current[0, 0])  # add f to rolling sum
        #
        # # return original database to storage
        # self.X = old_X
        # self.y = old_y
        # self.SI = old_sig
        # return np.log(sum) # edited by Max to scale down the objective function
        #
        # # for i in range(1,len(data)):
        # #     data_now = data[0:i]
        # #     self.fit(data_now[0],data_now[1])
        # #     F += self.f(x)
        # # return F

