__author__ = 'p2admin'
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import numpy as np
sns.set_style('ticks');
sns.set_palette('Set1')
import json
from pymc3 import *
import Preprocess
import Kriging
import CovarianceEstimate

pre = Preprocess(all_dat='../all_games.pkl', pca_model='../eco_full_pca.pkl')
X, y = pre.ready_player_one(2)
scale = MinMaxScaler((-1.,1.))
X = scale.fit_transform(X)

from tqdm import tqdm

file_address = 'p2_bfgs_sigma_alpha8.286TRUNCATED.json'
with open(file_address, 'r') as f:
    best_obj, best_sig = json.load( f)
f.close()

# unit_sig = np.ones(31)
# bounds = np.array(31*[[-1., 1.]])
# bestKrig = Kriging(best_sig, bounds=bounds)
# bestKrig.fit(X,y)

bounds = np.array(30*[[-1., 1.]])
num_ini_guess = 2
kriging_model = Kriging(best_sig, bounds=bounds, num_ini_guess=num_ini_guess)

with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define likelihood
    likelihood = kriging_model.obj(l,a)

with model:
    start = {'Intercept_logodds': np.array(-1.4488360894175776),
             'sigma_log': np.array(-0.7958719902826098),
             'x_logodds': np.array(-0.3564261325183015)}
    step = NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = sample(2000, step, start=start, progressbar=True) # draw 2000 posterior samples using NUTS sampling