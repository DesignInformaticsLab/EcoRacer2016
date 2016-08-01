__author__ = 'p2admin'
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import pinv2, inv
from pyDOE import lhs
from scipy.misc import logsumexp
import mcint
from scipy.stats import norm

def importancesampling(f, guess, sample_size):
    # use importance sampling with a normal distribution
    # use two scales of normal for local and global

    scale1 = np.sqrt(1./2./a)
    scale2 = np.sqrt(1./2./a)
    # C = norm.cdf(1., loc=0.5, scale=scale1) - norm.cdf(0., loc=0.5, scale=scale1)
    C = 1.
    A = []
    samples1 = np.random.normal(guess, scale1, size=(sample_size/2,1))
    samples2 = np.random.normal(guess, scale2, size=(sample_size/2,1))
    samples3 = np.random.uniform(size=(sample_size/2,1))
    samples3 = samples3*(bounds[1]-bounds[0])+bounds[0]
    W1 = 1./2./(scale1**2.)*(guess-samples1)**2.
    W2 = 1./2./(scale2**2.)*(guess-samples2)**2.
    for x in samples1:
        # A.append(f(x)/(C+np.exp(-(x-guess)**2./2./(scale1**2.))/np.sqrt(2*np.pi)/scale1)/sample_size*2*C)
        A.append(np.log(f(x))-np.log(1+np.exp(-(x-guess)**2./2./(scale1**2.))/np.sqrt(2.*np.pi)/scale1)
                     -np.log(sample_size/2.))
    for x in samples3:
        # A.append(f(x)/(C+np.exp(-(x-guess)**2./2./(scale1**2.))/np.sqrt(2*np.pi)/scale1)/sample_size*2*C)
        A.append(np.log(f(x))-np.log(1+np.exp(-(x-guess)**2./2./(scale1**2.))/np.sqrt(2.*np.pi)/scale1)
                     -np.log(sample_size/2.))
    # return sum(A)
    return logsumexp(A)

b = 1000.
a = 1e9
f = lambda x: 1+b*np.exp(-a*(x-0.5)**2.)
# f = lambda x: 0.
sample_size = 10000
x0 = 0.5
bounds = np.array([0., 1])


def integrand(x):
    return b*np.exp(-a*(x-0.5)**2.)

def sampler():
    while True:
        x = np.random.uniform(size=(1,1))
        x = x*(bounds[1]-bounds[0])+bounds[0]
        yield (x)
domainsize = 1
nmc = sample_size
# I = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0][0]
I = np.exp(importancesampling(f, x0, sample_size))
print I

# import scipy.integrate as integrate
# result = np.log(integrate.quad(lambda x: 1+1e6*np.exp(-1e9*(x-0.5)**2), 0, 1))
# print result
print 1+b*np.sqrt(np.pi/a)*(norm.cdf(1., loc=0.5, scale=np.sqrt(1./2./a)) - norm.cdf(0., loc=0.5, scale=np.sqrt(1./2./a)))
