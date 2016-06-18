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

file_address = 'solution_obj_name_branin_maxiter_100_repeat_30.pkl'
with open(file_address, 'r') as f:
    dat = pickle.load(f)

solution = np.array(dat['solution'])
print solution.shape
print solution[0,0,1].shape

# DO NOT RUN UNLESS YOU HAVE A LONG TIME TO WAIT!
def max_likelihood_estimate(samples, guesses):

    NUM_SAMPLES = samples
    num_ini_guess = guesses

    from estimate_sigma import CovarianceEstimate

    sigs = ['0.01', '0.1', '1.0', '10.0']
    for no, label in tqdm(enumerate(sigs), desc='true lambda loop'):
        trials = np.zeros((30,4,4))
        # print 'now on true_sig = ', label

        # bounds = np.array([[-5, 5], [-5, 5], [-5, 5],
        #                    [-5, 5], [-5, 5], [-5, 5]])  # for rosenbrock-6dim
        # bounds = np.array([[-3,3],[-3,3]])  # for sixmin
        bounds = np.array([[-5, 10], [0, 15]])  # for branin
        # for sig_no in enumerate([0.01,0.1,1.0,10.]):

        for trial in trange(30, desc='trials loop'):
            # print 'now calculating trial #'+str(trial+1)
            solution_X = solution[trial, no, 0] # test sigma = 0.01
            solution_y = solution[trial, no, 1]

            ce = CovarianceEstimate(solution_X[:NUM_SAMPLES], solution_y[:NUM_SAMPLES], bounds, num_ini_guess)
            sig_scale = np.array([0.01, 0.1, 1., 10.])
            alpha_set = np.array([0.01, 0.1, 1., 10.])
            # alpha_set = np.array([1e-5, 1e-4, 1e-3, 1e-2])
            grid_result = np.zeros((sig_scale.shape[0], alpha_set.shape[0]))
            for i, s in enumerate(sig_scale):
                sig_inv = np.ones(bounds.shape[0])*s
                for j, alpha in enumerate(alpha_set):
            #         print j
                    grid_result[i,j] = ce.model.obj(sig_inv, alpha)
            trials[trial, :, :] = grid_result[:]
            # time.sleep(10)


        data = np.copy(trials)

        # Write the array to disk
        path = os.path.expanduser('branin_ML/sample_study/all'+label+'branin_{:d}iter_{:d}init.txt'.format(NUM_SAMPLES, num_ini_guess))
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


if __name__ == '__main__':
    """Call as script: <python ./all_30_trials_branin.py no_observations>"""
    import sys

    print(sys.argv)
    samples = int(sys.argv[1])
    guesses = int(sys.argv[1])
    for guess in trange(1, guesses, desc='No. init samples loop'):
        max_likelihood_estimate(samples, guess)
