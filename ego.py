__author__ = 'Thurston Sexton'
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import pinv2, inv

class Kriging():

    """
    This is the actual optimization class, that will interface with the higher level regression.
    """
    #def __init__(self, sig, X, y):
    def __init__(self, sig_inv):
        #self.X = X
        #self.y = y

        self.SI = np.diag(sig_inv)
        #self.Sigma = sig
        self.X = np.array([[]]) # observed inputs, column vars
        self.y = np.array([[]]) # observed scores (must be 2-D)


        # self.model = np.array([])

    #def fit(self):
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.R = self.R_ij(self.X)
        #print self.R
        self.RI = pinv2(self.R)
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
        den = ones.T.dot(self.R.dot(ones))
        return num/den

    def yhat(self, x):
        # the kriging surface for given X,y
        r = self.r_i(x, self.X)
        return self.b + r.T.dot(self.RI.dot(np.subtract(self.y, self.b)))

    def get_s(self, x):
        dim = np.size(self.y)
        ones = np.ones(dim)

        r = self.r_i(x, self.X)

        # WARNING: this is getting runtime warnings (invalid value encountered in divide)
        mse = 1-r.T.dot(self.RI.dot(r))+(1.-ones.T.dot(self.RI.dot(r)))**2/(1. - ones.T.dot(self.RI.dot(ones)))

        sig = np.sqrt((self.y-self.b).T.dot(self.RI.dot(self.y-self.b))/dim)

        # WARNING: this is getting runtime warnings (invalid value encountered in multiply/sqrt)
        return np.sqrt(mse)*sig

    def f(self, x):
        if x.size == self.X.shape[1]:
            x = np.array(x, ndmin=2) #ensure is 2D
        # expected improvement function, given the model y_h
        y_h = self.yhat(x)
        ymax = np.max(self.y)
        s = self.get_s(x)
        z = np.divide(np.subtract(ymax, y_h), s)
        pdf = norm.pdf(z)
        cdf = norm.cdf(z)
        f_x = np.multiply(np.subtract(ymax, y_h), pdf) + np.multiply(s, cdf)
        return f_x

    def obj(self, sig_inv):
        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]

        #self.Sigma = sig  # replace stored sigma with supplied
        self.SI = np.diag(sig_inv)
        sum = 0.  # initiate loop
        self.fit(old_X[:1],old_y[:1])  # first observation
        for i, x in enumerate(old_X, 1): #loop through the rest
            f_current = self.f(x)  # expected improvement for current obs.
            self.fit(old_X[:i], old_y[:i])  # fit up to current obs.
            sum += np.nan_to_num(f_current[0, 0])  # add f to rolling sum

        # return original database to storage
        self.X = old_X
        self.y = old_y
        return sum

        # for i in range(1,len(data)):
        #     data_now = data[0:i]
        #     self.fit(data_now[0],data_now[1])
        #     F += self.f(x)
        # return F

