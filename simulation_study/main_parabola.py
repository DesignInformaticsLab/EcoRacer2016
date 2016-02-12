__author__ = 'Max Yi Ren'

from ego_solver import EGO
import numpy as np
from matplotlib import colors, ticker, cm
import matplotlib.pyplot as plt
import pickle
import os

# branin function
# obj_name = 'branin'
# obj = lambda x1, x2: (x2-5./4./np.pi/np.pi*x1**2+5./np.pi*x1-6)**2 + 10.*(1-1./8/np.pi)*np.cos(x1) + 10.

# obj_name = 'sixmin'
# obj = lambda x, y: 4.*x**2 - 2.1*x**4 + x**6/3. + x*y + 4.*y**2 + 4.*y**4

# obj_name = 'rosenbrock-6dim'
# def obj(x1, x2, x3, x4, x5, x6):
#     x = np.array([x1, x2, x3, x4, x5, x6])
#     """The Rosenbrock function"""
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)  # from scipy.optimize

# Parabolic:
obj_name = 'parabolic'
obj = lambda x, y: x**2 + y**2

# sig_scale = np.array([0.01, 0.1, 1., 10.])


sig_scale = np.array([1.])
max_iter = 100

num_ini_guess = 10
# bounds = np.array([[-5, 10], [0, 15]]) #  for branin
bounds = np.array([[-3, 3], [-3, 3]])  # for sixmin
# bounds = np.array([[-5, 5], [-5, 5], [-5, 5],
#                    [-5, 5], [-5, 5], [-5, 5]])  # for rosenbrock-6dim
repeat = 30
extreme = 100.
file_address = './solution_obj_name_' + obj_name + '_maxiter_' + str(max_iter) + \
               '_repeat_' + str(repeat) + 'x'+str(extreme) + '.pkl'


if not os.path.isfile(file_address):
    solution = np.empty((repeat, sig_scale.shape[0]), object)
    for i in np.arange(repeat):
        for j, sigma_inv in enumerate(sig_scale):
            # sig_inv = np.ones(2)*sigma_inv
            sig_inv = np.ones(2)*sigma_inv
            sig_inv[0] = extreme
            solver = EGO(sig_inv, obj, bounds, max_iter, num_ini_guess)
            solution_X, solution_y = solver.solve()
            solution[i, j] = (solution_X, solution_y)

    # save the solution
    with open(file_address, 'w') as f:
        pickle.dump({'solution': solution.tolist(), 'extreme': extreme, 'obj_name': obj_name,
                     'max_iter': max_iter}, f)
    f.close()
else:
    with open(file_address, 'r') as f:
        data = pickle.load(f)
    f.close()
    solution = np.array(data['solution'])
    # sig_scale = np.array(data['sig_scale'])
    max_iter = data['max_iter']

    # solution_X = np.array(solution['X'])
    # solution_y = np.array(solution['y'])

## draw solution traj ##
# X1 = np.linspace(-5, 10, 100)
# X2 = np.linspace(0, 15, 100)
# x1, x2 = np.meshgrid(X1, X2)
# ytrue = obj(x1, x2)
# plt.contourf(x1, x2, ytrue, 100, cmap=cm.bone)
# plt.plot(solution_X[:, 0], solution_X[:, 1])
# plt.show()


# y_min_mean = np.zeros((sig_scale.shape[0],max_iter))
# y_min_std = np.zeros((sig_scale.shape[0],max_iter))
# y_min_temp = np.zeros((repeat, max_iter))
# for j, sigma_inv in enumerate(sig_scale):
#     for i in np.arange(repeat):
#         y = solution[i,j,1]
#         for k in np.arange(y.shape[0]):
#             y_min_temp[i, k] = np.min(y[:k+1])
#         y_min_temp[i, y.shape[0]:] = y_min_temp[i, y.shape[0]-1]
#     y_min_mean[j,:] = np.mean(y_min_temp, axis=0)
#     y_min_std[j,:] = np.std(y_min_temp, axis=0)
#     plt.errorbar(np.arange(0, max_iter), y_min_mean[j,:], yerr=y_min_std[j,:], fmt='-o')
# plt.show()

# solution_X = solution[0,2,0] # test sigma = 0.1
# solution_y = solution[0,2,1]
# from estimate_sigma import CovarianceEstimate
# ce = CovarianceEstimate(solution_X, solution_y, bounds, num_ini_guess)
# sig_scale = np.array([0.01, 0.1, 1., 10.])
# alpha_set = np.array([0.01, 0.1, 1., 10., 100., 1000.])
# grid_result = np.zeros((sig_scale.shape[0], alpha_set.shape[0]))
# for i, s in enumerate(sig_scale):
#     sig_inv = np.ones(2)*s
#     for j, alpha in enumerate(alpha_set):
#         grid_result[i,j] = ce.model.obj(sig_inv, alpha)
#
# wait = 1
# f, best_sigma = ce.solve()

# file_address = './estimated_sigma.pkl'
# if not os.path.isfile(file_address):
#     # save the solution
#     with open(file_address, 'w') as f:
#         pickle.dump(best_sigma.tolist(), f)
#     f.close()