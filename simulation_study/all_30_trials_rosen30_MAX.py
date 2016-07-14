__author__ = 'p2admin'
__author__ = 'Thurston'

from ego_solver import EGO
import numpy as np
import os
from tqdm import trange, tqdm
# from matplotlib import colors, ticker, cm
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import pickle
import time
from pyDOE import lhs
import random
from estimate_sigma import CovarianceEstimate

file_address = 'solution_obj_name_rosenbrock-30dim_maxiter_100_repeat_30.pkl'
with open(file_address, 'r') as f:
    dat = pickle.load(f)

solution = np.array(dat['solution'])
print solution.shape
print solution[0,0,1].shape

num_trial = 30

np.random.seed(0)
sample = np.random.uniform(size=(100,30))

# DO NOT RUN UNLESS YOU HAVE A LONG TIME TO WAIT!
def max_likelihood_estimate(samples):
    sigs = ['0.1','1.0','10.0']
    sig_scale = np.array([0.01, 0.1, 1., 10.])
    alpha_set = np.array([0.01, 0.1, 1., 10.])
    NUM_SAMPLES = samples
    num_ini_guess = 2
    bounds = np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                        [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                        [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                        [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
                        [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]])  # for rosenbrock-30dim


    for no, label in enumerate(sigs):
        grid_result = np.zeros((sig_scale.shape[0], alpha_set.shape[0], NUM_SAMPLES-num_ini_guess,num_trial))

        for trial in range(num_trial):
            solution_X = solution[trial, no, 0]
            solution_y = solution[trial, no, 1]

            ce = CovarianceEstimate(solution_X[:NUM_SAMPLES], solution_y[:NUM_SAMPLES], bounds, num_ini_guess, sample)

            for i, s in enumerate(sig_scale):
                sig_inv = np.ones(bounds.shape[0])*s
                for j, alpha in enumerate(alpha_set):
                    temp = ce.model.obj(sig_inv, alpha)
                    if temp.shape[0]<NUM_SAMPLES-num_ini_guess:
                        temp = np.hstack((temp,[np.nan]*(NUM_SAMPLES-num_ini_guess-temp.shape[0])))
                    grid_result[i,j,:,trial] = temp

        for guess in range(20,NUM_SAMPLES):
            trials = np.zeros((num_trial,4,4))
            for trial in range(num_trial):
                trials[trial, :, :] = grid_result[:,:,guess-num_ini_guess,trial]

            data = np.copy(trials)
            # Write the array to disk
            path = os.path.expanduser('rosen30_ML/all'+label+'rosen30_{:d}iter_{:d}init.txt'.format(NUM_SAMPLES, guess))
            with file(path, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                outfile.write('# ' +label+'sig - Array shape (trial/sig/alpha): {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                for data_slice in data:

                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.
                    np.savetxt(outfile, data_slice, fmt='%.4e')

                    # Writing out a break to indicate different slices...
                    outfile.write('# New trial\n')
                # time.sleep(10)

samples = 50
# guesses = 24
# for guess in trange(11, guesses, desc='No. init samples loop'):
max_likelihood_estimate(samples)