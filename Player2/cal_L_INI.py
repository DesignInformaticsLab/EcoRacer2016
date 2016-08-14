__author__ = 'Thurston Sexton'
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json


from ego import Kriging
from preprocess import Preprocess
from reg import CovarianceEstimate
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from pyDOE import lhs
from scipy.misc import logsumexp
import scikits.bootstrap as boot
# #
# # get data from the game
# # delete the parameters if performing first-time or new player.
# # Parameters are there to speed up after saving a pkl.
pre = Preprocess(pca_model='../eco_full_pca.pkl', all_dat='../all_games.pkl')
# pre = Preprocess()
# pre.get_json('../alluser_control.json')  # uncomment this to create the pkl file needed!!
# pre.train_pca()
X, y = pre.ready_player_one(2) # MAX: first dimension is number of plays, second is solution space dimension

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)

total_no_iters = 31
n_trial = 1
dim = 31
bounds = np.array([[-1.,1.]]*31)
p = np.zeros((total_no_iters-2,))+0.0 # initialize
solution_XX = X
E_true = np.array([squareform(pdist(solution_XX[:i,:]))[i-1,:(i-2)].min() for i in range(3,total_no_iters+1)])
alpha = 0.01
for n,tr in enumerate(range(3,total_no_iters+1)):
    true_set = solution_XX[:tr,:]
    samples = np.random.uniform(size=(10000,dim))
    samples = samples*(bounds[:, 1]-bounds[:, 0])+bounds[:, 0]
    E_samp = (cdist(true_set[:-1],samples).min(axis=0))
    log_prob = alpha*E_true[n] - logsumexp(alpha*E_samp - np.log(10000))
    p[n] = -log_prob

print p