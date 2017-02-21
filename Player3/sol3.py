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

# #
# # get data from the game
# # delete the parameters if performing first-time or new player.
# # Parameters are there to speed up after saving a pkl.
pre = Preprocess(pca_model='../eco_full_pca.pkl', all_dat='../all_games.pkl')
# pre = Preprocess()
# pre.get_json('../alluser_control.json')  # uncomment this to create the pkl file needed!!
# pre.train_pca()
X, y = pre.ready_player_one(3)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)

########
n_trajectory = 71
X, y = X[:n_trajectory], y[:n_trajectory] # only use the first few plays
########

# # get sigma estimate that maximizes the sum of expected improvements
bounds = np.array(31*[[-2., 1.]])
xbounds = np.array(31*[[-1., 1.]])

initial_guess = np.log10(np.array(31*[1.]))

# below are the pre-calculated log likelihood (not negative!) values for l_INI, with alpha_INI = 10, 1, 0.01
# the more positive these values are, the more likely the sample follows max-min sampling

from cal_L_INI import cal_L_INI

l_INI_10 = -cal_L_INI(10.)
l_INI_1 = cal_L_INI(1.)
l_INI_001 = -cal_L_INI(.01)



sample_size = 100
num_ini_guess = 2
alpha = 10.0
soln = CovarianceEstimate(X, y, bounds=bounds, xbounds=xbounds, alpha=alpha, sample_size=sample_size,
                          num_ini_guess=num_ini_guess, initial_guess=initial_guess, l_INI=l_INI_10[:(n_trajectory-2)])
# x_temp =np.random.normal(initial_guess, scale=0.1, size=(1,31))
# # x_temp = np.ones((31,))*10.0
f0 = soln.model.obj(initial_guess, alpha=alpha, l_INI=l_INI_10[:(n_trajectory-2)])
print f0
# sig_test = np.zeros(31)
# sig_test[-1] = 2.6
# soln.model.f_path(sig_test)
[obj_set, sigma_set] = soln.solve(plot=False)

# # pick the best solution
obj = obj_set.min(axis=0)
sigma = sigma_set[obj_set.argmin(axis=0), :]
print obj, sigma

# # load bounds
# from numpy import loadtxt
# bounds = loadtxt("ego_bounds.txt", comments="#", delimiter=",", unpack=False)
#
# # store sigma for simulation
# # TODO: need to specify file name based on settings, e.g., optimization algorithm and input data source (best player?)

file_address = 'p3_bfgs_sigma_alpha'+str(soln.alpha)+'_0220_sample100_aroundx1_first71.json'
# x0: thurston optimal for 31 plays
# x1: thurston optimal for 12 plays
# x2: thurston optimal for 5 plays

with open(file_address, 'wb') as f:
    # pickle.dump([obj_set, sigma_set], f)
    json.dump([obj, sigma.tolist()], f, sort_keys=True, indent=4, ensure_ascii=False)
f.close()

# store all pcs to a json
# from sklearn.externals import joblib
# temp = joblib.load('../eco_full_pca.pkl')
#
# file_address = 'ica.json'
# with open(file_address, 'w') as f:
#     json.dump(temp.components_.tolist(), f, sort_keys=True, indent=4, ensure_ascii=False)
# f.close()


with open('p3_range_transform.json', 'w') as outfile:
    json.dump({'range':scale.scale_.tolist(), 'min':scale.min_.tolist()},
              outfile, sort_keys=True, indent=4, ensure_ascii=False)
with open('p3_ICA_transform.json', 'w') as outfile:
    json.dump({'mix':pre.pca.mixing_.tolist(), 'unmix':pre.pca.components_.tolist(), 'mean':pre.pca.mean_.tolist()},
              outfile, sort_keys=True, indent=4, ensure_ascii=False)

np.savetxt('mix_scaled_p3_initplay.txt', X[:num_ini_guess])  # first two plays for later init.

# A = pre.pca.components_
# Std_inv = np.diag(1/scale.std_)
# vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
# np.savetxt('visualize_this.txt', vis)