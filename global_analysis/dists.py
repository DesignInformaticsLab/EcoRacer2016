__author__ = 'Thurston'

from ego import Kriging
from preprocess import Preprocess
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pre = Preprocess(pca_model='../eco_full_pca.pkl', all_dat='../all_games.pkl')
# pre = Preprocess()
# pre.get_json('../alluser_control.json')  # uncomment this to create the pkl file needed!!
# pre.train_pca()
X, y = pre.ready_player_one(2)

unit_sig = np.ones(31)

# scale = StandardScaler()
scale = MinMaxScaler((-1., 1.))
X = scale.fit_transform(X)
#
# # get sigma estimate that maximizes the sum of expected improvements

import scipy.optimize as opt
all_sigs = np.zeros((len(pre.full_tab['id'].tolist()), 31))
all_improv = np.zeros_like(pre.full_tab['id'].tolist())
lb = 0.01
ub = 100.
bounds = [(lb, ub)]*31

for n, i in enumerate(pre.full_tab['id'].tolist()):

    a, b = pre.prep_by_id(i)
    print i, len(a)
    if len(a) == 1 or len(a) == 0:
        all_improv[n] = 0
        continue
    krig = Kriging(unit_sig)
    krig.fit(a, b)
    x0 = np.ones(31)
    func = lambda x: - krig.f_by_sig(x)
    res = opt.minimize(func, x0=x0, bounds=bounds, method='SLSQP', tol=1e-5,
                               options = {'eps': 1e-2, 'iprint': 2, 'disp': True, 'maxiter': 100})
#     all_improv[n] = np.nan_to_num(krig.f(a[-1])[0])
    all_improv[n] = res.fun
    all_sigs[n] = res.x
    print res.fun, res.x

np.savetxt('opt_f_allplays.txt', all_improv)  # first two plays for later init.
np.savetxt('opt_sig_allplays.txt', all_sigs)
