__author__ = 'Thurston'

import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, cdist, squareform
from pyDOE import lhs
from scipy.misc import logsumexp


pal = sns.color_palette('Dark2', n_colors=2, desat=.6)
sns.set_palette(pal)
sns.set_context(context='paper', font_scale=1.5)
sns.set_style('ticks')

from sklearn.preprocessing import MinMaxScaler


def cuml_like(arr1, arr2):
    arr1=MinMaxScaler().fit_transform(arr1.astype(float).reshape(-1,1))
    arr2=MinMaxScaler().fit_transform(arr2.astype(float).reshape(-1,1))
    if arr1.size!=arr2.size:
        raise Exception('must be equal-sized arrays arr1 and arr2')
    new = np.zeros_like(arr1)
    for n in range(new.size):
        new[n] = arr1[:n+1].sum()+arr2[n+1:].sum()
    return new


def get_Ls(lam, iters):
    inits = np.arange(1,iters)
    Ls = np.zeros((inits.size,4))
    for n,i in enumerate(inits):
        fname='./branin_ML/sample_study/all{}branin_{}iter_{:d}init.txt'.format(lam,iters,i)
        data = np.loadtxt(fname).reshape((30,4,4))
        Ls[n] = -data.mean(axis=0).min(axis=0)
    return Ls


from tqdm import trange
file_address = 'solution_obj_name_branin_maxiter_100_repeat_30.pkl'
with open(file_address, 'r') as f:
    dat = pickle.load(f)

names=['0.01','0.1','1.0','10.0']

df = pd.DataFrame()

for lam_no in trange(4, desc='true lambda loop'):
    all_df = pd.DataFrame()  # dataframe for this specific true lambda
    for no_iters in trange(5, 26, desc='observation no loop'):
        mins_df = pd.DataFrame(index=np.arange(30), columns=names)
#         no_iters = 20
        true_lam = lam_no
        l_ego = get_Ls(names[true_lam], no_iters)[1:]

        # dat['solution'][true_lam, coord_vs_ei, no_samples]
        solution_X = np.dstack([i[true_lam][0][:no_iters,:].T for i in dat['solution']])
        solution_y = np.dstack([i[true_lam][1][:no_iters] for i in dat['solution']])

        E_true = np.array([[pdist(solution_X[:,:i,j].T).min() for i in range(2,25)] for j in range(30)])

        bounds = np.array([[-5, 10], [0, 15]])  # for branin
        t_samp = range(2,no_iters)
        p = np.zeros_like(t_samp)  # initialize
        alpha = 100.

        for j in range(4):  # the test-lambda
            mins = []  # for min-seeking statistics
            for trial in range(30):
                for n,tr in enumerate(t_samp):
                    true_set = solution_X[:,:tr, 0].T
                    bounds = np.array([[-5, 10], [0, 15]])  # for branin
                #     samples = lhs(2, 100)
                    samples = np.random.uniform(size=(10000,2))
                    samples = samples*(bounds[:, 1]-bounds[:, 0])+bounds[:, 0]
                    E_samp = (cdist(true_set[:-1],samples).min(axis=0))
                    log_prob = alpha*E_true[trial,n] - logsumexp(alpha*np.append(E_samp.T, E_true[trial,n]).T)
                    p[n] = -log_prob

                combined = cuml_like(p, l_ego[:, j])
        #
                mins += [np.min(combined)]
            mins_df[names[j]] = mins  # data-frame of this lambda's min-Likelihood's
        mins_df = pd.melt(mins_df, value_vars=names, var_name='lambda',value_name='min_L')  # make long-form
        mins_df['no_obsv'] = no_iters
        all_df = pd.concat([all_df, mins_df])  # add to overall data
    all_df['true_lam'] = names[lam_no]
#     print all_df.head()

    df = pd.concat([df,all_df])  # add to main database
#         sns.swarmplot(data=mins_df, x='lambda',y='mins', ax=ax)
sns.factorplot(data=df, x='no_obsv', y='min_L', hue='lambda', row='true_lam',
               kind='point', aspect=3, sharey=False)
df.to_csv('branin_likelihood_combined.csv')  # save main database
