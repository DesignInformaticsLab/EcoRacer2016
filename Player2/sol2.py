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
n_trajectory = 31
X, y = X[:n_trajectory], y[:n_trajectory] # only use the first few plays
########

# # get sigma estimate that maximizes the sum of expected improvements
bounds = np.array(31*[[-2., 1.]])
xbounds = np.array(31*[[-1., 1.]])

# initial_guess = np.array([0.0]*31)

initial_guess = np.log10(np.array([
        2.2948942550847264,
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
        0.09020801560634745


        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.7506890268438688,
        # 0.01,
        # 0.9340860245275615,
        # 0.01,
        # 0.256325554791345,
        # 3.307235852132785,
        # 0.01,
        # 100.0,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 10.949856064516513


        # 0.009999999999999995,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.6146608596420602,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.29922765226743187,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.2835921874053612,
        # 2.0037564403933827,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.9760224662309374,
        # 0.01,
        # 0.009999999999999995,
        # 0.01,
        # 0.7445227026771479,
        # 0.009999999999999995

        # this is from p2_bfgs_sigma_MLE_31.json
        # 0.01,
        # 0.72368274254733056,
        # 3.1259148454693704,
        # 0.27991899822525201,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 1.0570312287860779,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.16075275891036223,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 0.01,
        # 3.8464362011368398
    ]))

# below are the pre-calculated log likelihood (not negative!) values for l_INI, with alpha_INI = 10, 1, 0.01
# the more positive these values are, the more likely the sample follows max-min sampling
l_INI_10 = -np.array([ 14.269769,    18.51242514,  15.43307011,  14.73611888,  10.11151815,
  10.38230399,   6.11636895 , 17.87209467 , 11.28252325,  16.09211475,
  19.13666231,  14.26350805,  20.88717118 , 17.97391579,  18.05726332,
  18.93868602,  13.05189466 , 13.90962204 , 23.3297502 ,  12.40313786,
  18.89782523 , 12.11301276 , 20.62505659 , 19.62516373 , 13.25618674,
  14.86894299 , 16.71269039  , 9.61347853 ,  7.16914187])
l_INI_1 = -np.array([ 0.86115837,  1.22677822,  0.92775171,  0.9136831,   0.44296968,  0.50790001,
  0.15516951,  1.22674106,  0.65270824,  1.16261199,  1.38351062,  0.98991564,
  1.62832714,  1.33700206,  1.32064881,  1.43865368,  0.84637371,  0.96145962,
  1.86322818,  0.76696031,  1.39329787,  0.75810048,  1.59063005,  1.49758157,
  0.86818537,  1.01713967,  1.21994997,  0.50326903,  0.26969321])
l_INI_001 = -np.array([ 0.0077654,   0.01148891,  0.0085491,   0.00846845,  0.00377641,  0.00445976,
  0.00094088,  0.01169417,  0.00588484,  0.01099119,  0.01322844,  0.00925459,
  0.0157014,   0.01277546,  0.01259828,  0.0137948,   0.00783676,  0.00910085,
  0.01805538,  0.00703961,  0.01333193,  0.0069961,   0.01534839,  0.01438826,
  0.00808909,  0.00955123,  0.01161055,  0.00448607,  0.00210488])



sample_size = 1000
num_ini_guess = 2
alpha = 10.0
soln = CovarianceEstimate(X, y, bounds=bounds, xbounds=xbounds, alpha=alpha, sample_size=sample_size,
                          num_ini_guess=num_ini_guess, initial_guess=initial_guess, l_INI=l_INI_10[:(n_trajectory-2)])
# x_temp =np.random.normal(initial_guess, scale=0.1, size=(1,31))
# # x_temp = np.ones((31,))*10.0
f0 = soln.model.obj(initial_guess, alpha=alpha, l_INI=l_INI_10[:(n_trajectory-2)])
print f0
# # sig_test = np.zeros(31)
# # sig_test[-1] = 2.6
# # soln.model.f_path(sig_test)
[obj_set, sigma_set] = soln.solve(plot=False)
#
# # pick the best solution
obj = obj_set.min(axis=0)
sigma = sigma_set[obj_set.argmin(axis=0), :]
print obj, sigma
#
# # # load bounds
# # from numpy import loadtxt
# # bounds = loadtxt("ego_bounds.txt", comments="#", delimiter=",", unpack=False)
# #
# # # store sigma for simulation
# # # TODO: need to specify file name based on settings, e.g., optimization algorithm and input data source (best player?)
#
file_address = 'p2_bfgs_sigma_alpha'+str(soln.alpha)+'_0219_sample1000_aroundx0.json'
# # x0: thurston optimal for 31 plays
# # x1: thurston optimal for 12 plays
# # x2: thurston optimal for 5 plays
# # x4: for test only
#
with open(file_address, 'wb') as f:
    # pickle.dump([obj_set, sigma_set], f)
    json.dump([obj, sigma.tolist()], f, sort_keys=True, indent=4, ensure_ascii=False)
f.close()
#
# # store all pcs to a json
# # from sklearn.externals import joblib
# # temp = joblib.load('../eco_full_pca.pkl')
# #
# # file_address = 'ica.json'
# # with open(file_address, 'w') as f:
# #     json.dump(temp.components_.tolist(), f, sort_keys=True, indent=4, ensure_ascii=False)
# # f.close()
#
#
# # with open('p2_range_transform.json', 'w') as outfile:
# #     json.dump({'range':scale.scale_.tolist(), 'min':scale.min_.tolist()},
# #               outfile, sort_keys=True, indent=4, ensure_ascii=False)
# # with open('p2_ICA_transform.json', 'w') as outfile:
# #     json.dump({'mix':pre.pca.mixing_.tolist(), 'unmix':pre.pca.components_.tolist(), 'mean':pre.pca.mean_.tolist()},
# #               outfile, sort_keys=True, indent=4, ensure_ascii=False)
#
# # np.savetxt('mix_scaled_p2_init_ALL5.txt', X)  # first two plays for later init.
#
# # A = pre.pca.components_
# # Std_inv = np.diag(1/scale.std_)
# # vis = A.T.dot(Std_inv.dot(np.diag(sigma).dot(Std_inv.dot(A))))
# # np.savetxt('visualize_this.txt', vis)