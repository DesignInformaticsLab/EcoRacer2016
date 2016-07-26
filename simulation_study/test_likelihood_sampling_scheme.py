from ego_solver import EGO
import numpy as np
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from pyDOE import lhs
import random
from estimate_sigma import CovarianceEstimate

file_address = 'solution_obj_name_rosenbrock-30dim_maxiter_100_repeat_30.pkl'
# file_address = 'solution_obj_name_branin_maxiter_100_repeat_30.pkl'
with open(file_address, 'r') as f:
    dat = pickle.load(f)
solution = np.array(dat['solution'])
num_trial = 30

# np.random.seed(0)
sample_size = 1000

sig_guess = 0.1
alpha = 1.0

num_ini_guess = 10

# bounds = np.array([[-5, 10], [0, 15]]) #  for branin
bounds = np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]])  # for rosenbrock-30dim

NUM_SAMPLES = 20
method = 'mcmc'
single = False

solution_X = solution[0, 1, 0] #trial 0 (first arg), sigs = 0.1 (second arg)
solution_y = solution[0, 1, 1]
ce = CovarianceEstimate(solution_X[:NUM_SAMPLES], solution_y[:NUM_SAMPLES], bounds, num_ini_guess)
sig_inv = np.ones(bounds.shape[0])*sig_guess
grid_result = ce.model.obj(sig_inv, alpha, method, single, sample_size)+np.log(sample_size)

print -grid_result
