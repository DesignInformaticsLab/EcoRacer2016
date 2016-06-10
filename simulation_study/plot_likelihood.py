from ego_solver import EGO
import numpy as np
from matplotlib import colors, ticker, cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import time
pal = sns.color_palette('Dark2', n_colors=2, desat=.6)
sns.set_palette(pal)
sns.set_context(context='paper', font_scale=1.5)
sns.set_style('ticks')


sig_scale = np.array([0.01, 0.1, 1., 10.])
alpha_set = np.array([0.01, 0.1, 1., 10.])
# alpha_set = np.array([1e-5, 1e-4, 1e-3, 1e-2])

# data = np.loadtxt('all_10_trials_branin.txt').reshape((30,4,4))

c_m = sns.cubehelix_palette(as_cmap=True, light=1)

norm = colors.LogNorm(vmin=10., vmax=1e5)
s_m = cm.ScalarMappable(cmap=c_m, norm=norm)
#     norm.autoscale(X)
s_m.set_array([])


import matplotlib.colors as colors
import matplotlib.cm

names=['0.01','0.1','1.0','10.0']


file_address = 'solution_obj_name_branin_maxiter_100_repeat_30.pkl'
with open(file_address, 'r') as f:
    dat = pickle.load(f)

solution = np.array(dat['solution'])
print solution.shape
print solution[0,0,1].shape


from scipy.spatial.distance import pdist, cdist, squareform
from pyDOE import lhs
from scipy.misc import logsumexp

trial=0
solution_X = solution[trial, 1, 0] # test sigma = 0.01
solution_y = solution[trial, 1, 1]
# plt.plot(*solution_X[:10].T, ls='--', marker='o')

bounds = np.array([[-5, 10], [0, 15]])  # for branin

E_true = np.array([pdist(solution_X[:i]).min() for i in range(2,20)])
# plt.plot(range(2,20),np.exp(E_true))

t_samp = range(2,20)
p = np.zeros_like(t_samp)
E_true = np.array([pdist(solution_X[:i]).min() for i in t_samp])
cdist
alpha = 10.
for test in range(10):
    for n,tr in enumerate(t_samp):
        true_set = solution_X[:tr]

        bounds = np.array([[-5, 10], [0, 15]])  # for branin
    #     samples = lhs(2, 100)
        samples = np.random.uniform(size=(10000,2))
        samples = samples*(bounds[:, 1]-bounds[:, 0])+bounds[:, 0]

        E_samp = (cdist(true_set[:-1],samples).min(axis=0))
#         print np.append(E_samp.T, E_true[n]).shape
        log_prob = alpha*E_true[n] - logsumexp(alpha*np.append(E_samp.T, E_true[n]).T)
        p[n] = -log_prob
        plt.plot(t_samp,p, alpha=.1, color='k', ls=':')
#     plt.scatter(*samples.T)
# sns.distplot(min_dists)
# plt.plot(t_samp,p)
plt.show()



# import scikits.bootstrap as boot
# lo_hi = np.zeros((201,2))
# for i in range(201):
#      lo_hi[i] = boot.ci(mult[mult['Play No.']==i]['curr_best'].values, np.mean, alpha=.32)
# lo_hi