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
X, y = pre.ready_player_one(2)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)

########
# X, y = X[:12], y[:12]
########

# # get sigma estimate that maximizes the sum of expected improvements
bounds = np.array(31*[[-1., 1.]])

initial_guess = np.array([
        0.01,
        0.01,
        0.042309624581465505,
        0.3596366606394387,
        0.01,
        2.7829884240773324,
        0.15192136346511198,
        0.01,
        0.01,
        0.01,
        0.45266520067595917,
        0.01,
        0.01,
        0.01,
        0.12430166802060787,
        0.01,
        1.1314614190397028,
        0.48300704908795566,
        0.01,
        0.1197386464706936,
        0.01,
        0.01,
        1.4709645458255112,
        0.01,
        0.2905627718390898,
        0.04111797446313845,
        3.2457907227927465,
        0.01,
        0.01,
        0.09020801560634745,
        2.2948942550847264
    ])
sample_size = 1000
num_ini_guess = 2
soln = CovarianceEstimate(X, y, bounds=bounds, alpha=10.0, sample_size=sample_size,
                          num_ini_guess=num_ini_guess, initial_guess=initial_guess)
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

file_address = 'p2_bfgs_sigma_alpha'+str(soln.alpha)+'_0811.json'
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


# with open('p2_range_transform.json', 'w') as outfile:
#     json.dump({'range':scale.scale_.tolist(), 'min':scale.min_.tolist()},
#               outfile, sort_keys=True, indent=4, ensure_ascii=False)
# with open('p2_ICA_transform.json', 'w') as outfile:
#     json.dump({'mix':pre.pca.mixing_.tolist(), 'unmix':pre.pca.components_.tolist(), 'mean':pre.pca.mean_.tolist()},
#               outfile, sort_keys=True, indent=4, ensure_ascii=False)

# np.savetxt('mix_scaled_p2_init_ALL5.txt', X)  # first two plays for later init.

# A = pre.pca.components_
# Std_inv = np.diag(1/scale.std_)
# vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
# np.savetxt('visualize_this.txt', vis)