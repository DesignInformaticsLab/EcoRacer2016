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
X, y = pre.ready_player_one(1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)
#
# # get sigma estimate that maximizes the sum of expected improvements
soln = CovarianceEstimate(X, y)
# sig_test = np.zeros(31)
# sig_test[-1] = 2.6
# soln.model.f_path(sig_test)
[obj_set, sigma_set] = soln.solve(plot=True)

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

file_address = 'p2_slsqp_sigma.json'
with open(file_address, 'w') as f:
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


with open('p2_range_transform.json', 'w') as outfile:
    json.dump({'range':scale.scale_.tolist(), 'min':scale.min_.tolist()},
              outfile, sort_keys = True, indent = 4, ensure_ascii=False)
with open('p2_ICA_transform.json', 'w') as outfile:
    json.dump({'mix':pre.pca.mixing_.tolist(), 'unmix':pre.pca.components_.tolist(), 'mean':pre.pca.mean_.tolist()},
              outfile, sort_keys = True, indent = 4, ensure_ascii=False)

np.savetxt('mix_scaled_p1_0.txt', X[0])
# A = pre.pca.components_
# Std_inv = np.diag(1/scale.std_)
# vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
# np.savetxt('visualize_this.txt', vis)