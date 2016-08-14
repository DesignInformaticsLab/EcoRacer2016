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
    def __init__(self, sig_inv, bounds, num_ini_guess=2, sample_size=10000):
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
        self.size = np.prod(self.bounds[:, 1]-self.bounds[:, 0])# domain size
        self.sample_size = sample_size
        # setup random samples to calculate mean of expected improvement
        # self.samples = lhs(2, 100)  # for 2-dim funcs
        # self.samples = lhs(31, 500)  # for 6-dim rosenbrock
        # self.samples = self.samples*(self.bounds[:, 1]-self.bounds[:, 0])+self.bounds[:, 0]


    #def fit(self):
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.R = self.R_ij(self.X)
        #print self.R
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

        # WARNING: this is getting runtime warnings (invalid value encountered in multiply/sqrt)
        return np.sqrt(mse)*sig

    def f(self, x):
        if x.size == self.X.shape[1]:
            x = np.array(x, ndmin=2) #ensure is 2D
        # expected improvement function, given the model y_h

        y_h = self.yhat(x)
        ymax = np.max(self.y)
        s = self.get_s(x)
        z = np.divide(np.subtract(y_h, ymax), s+1e-12)
        if s == 0:
            pdf = 0
        else:
            pdf = norm.pdf(z)
        cdf = norm.cdf(z)
        f_x = np.multiply(np.subtract(y_h, ymax), cdf) + np.multiply(s, pdf)
        return f_x

    def f_path(self, sig_inv):
        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]
        old_sig = self.SI[:]

        #self.Sigma = sig  # replace stored sigma with supplied
        self.SI = np.diag(sig_inv)

        path = np.zeros(self.X.shape[0]-self.num_ini_guess)
        # print self.n, path.shape
        self.fit(old_X[:self.num_ini_guess],old_y[:self.num_ini_guess])  # first observation
        for i, x in enumerate(old_X[self.num_ini_guess:], self.num_ini_guess):

            path[i-self.num_ini_guess] = self.f(x)
            self.fit(old_X[:i+1], old_y[:i+1])

        # return original database to storage
        self.X = old_X
        self.y = old_y
        self.SI = old_sig

        self.recent_path = path[:]
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

    def broadcast_f_path(self, sig_inv, samples):
        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]
        old_sig = self.SI[:]

        # self.Sigma = sig  # replace stored sigma with supplied
        self.SI = np.diag(sig_inv)

        sample_size = samples.shape[0]
        sampled_path = np.zeros((self.n, sample_size))
        # print self.n, path.shape
        self.fit(old_X[:self.num_ini_guess], old_y[:self.num_ini_guess])  # first observation
        for i, x in enumerate(old_X[self.num_ini_guess:], self.num_ini_guess):
            sampled_path[i - 1] = np.apply_along_axis(self.f, 1, self.samples).T
            # for j, xx in enumerate(samples):
            #     sampled_path[i - 1, j] = self.f(xx)
            self.fit(old_X[:i + 1], old_y[:i + 1])

        # return original database to storage
        self.X = old_X
        self.y = old_y
        self.SI = old_sig
        return sampled_path

    def obj(self, log_sig_inv, alpha, l_INI):
        sig_inv = 10**log_sig_inv
        path = self.f_path(sig_inv)
        # sampled_path = self.sampled_f_path(sig_inv, self.samples)
        # # log_prob = np.log(1./(1.+np.sum(np.exp(alpha*(sampled_path.T - path)), axis=0)))
        # # sum_improv = np.sum(self.recent_path)
        # log_prob = alpha*path - logsumexp(alpha*np.vstack((sampled_path.T, path)).T, axis=1)

        zz = self.mcmc_f_path(alpha, sig_inv, self.sample_size)
        log_prob = alpha*path - zz

        L = self.cuml_like(l_INI, log_prob) # combine L_INI and L_BO
        # L = log_prob

        self.recent_path = log_prob
        sum_improv = np.sum(L)

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
    def mcmc_f_path(self, alpha, sig_inv, sample_size):
        domainsize = self.size
        self.SI = np.diag(sig_inv)

        # save whole database as a copy
        old_X = self.X[:]
        old_y = self.y[:]
        old_sig = self.SI[:]

        sampled_path = np.zeros(self.X.shape[0]-self.num_ini_guess)
        # print self.n, path.shape
        self.fit(old_X[:self.num_ini_guess],old_y[:self.num_ini_guess])  # first observation
        for i, x in enumerate(old_X[self.num_ini_guess:], self.num_ini_guess):
            result = self.importancesampling(x,sample_size,alpha,domainsize)
            sampled_path[i-self.num_ini_guess] = result
            self.fit(old_X[:i+1], old_y[:i+1])

        # return original database to storage
        self.X = old_X
        self.y = old_y
        self.SI = old_sig
        return sampled_path

    def importancesampling(self, guess, sample_size, alpha, domainsize):
        # use importance sampling with a normal distribution
        # use two scales of normal for local and global
        scale = 1e-1
        A = []
        temp = []
        self.samples1 = np.random.normal(guess, scale, size=(sample_size/2,self.p))
        self.samples3 = np.random.uniform(size=(sample_size/2,self.p))
        self.samples3 = self.samples3*(self.bounds[:, 1]-self.bounds[:, 0])+self.bounds[:, 0]
        for x in self.samples1:
            temp.append(self.f(x))
            A.append(self.f(x)*alpha-np.log(1+domainsize*np.exp(-np.linalg.norm(x-guess)**2./2./(scale**2.))/np.sqrt(2.*np.pi)/(scale**self.p))
                     -np.log(sample_size/2.))
        for x in self.samples3:
            temp.append(self.f(x))
            A.append(self.f(x)*alpha-np.log(1+domainsize*np.exp(-np.linalg.norm(x-guess)**2./2./(scale**2.))/np.sqrt(2.*np.pi)/(scale**self.p))
                     -np.log(sample_size/2.))
        return logsumexp(A)

    def cuml_like(self, arr1, arr2):
        if arr1.size!=arr2.size:
            raise Exception('must be equal-sized arrays arr1 and arr2')
        new = np.zeros(arr1.size+1)
        for n in range(new.size):
            new[n] = arr1[:n].sum()+arr2[n:].sum()
        new[arr1.size] = arr1.sum()
        return np.max(new)