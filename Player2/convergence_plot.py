__author__ = 'Thurston'
import sys
sys.path.insert(0, '../')

from ego import Kriging
from preprocess import Preprocess
import numpy as np
# from matplotlib import colors, ticker, cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import time
# %matplotlib inline
sns.set_style('ticks');
pal = sns.color_palette('Dark2', n_colors=2, desat=.6)
sns.set_palette(pal)
sns.set_context(context='paper', font_scale=1.5)

with open('simulation_04042016.npy', 'rb') as f:
    maxD = np.load(f)
print len(maxD[2][0])

"""Starting with 5 as initial, train on 5"""
score_5p = np.array([np.array(maxD[0][j]) for j in range(len(maxD[0]))])
mean_5p = np.mean(score_5p, axis = 0)
std_5p = np.std(score_5p, axis = 0)
score_5u = np.array([np.array(maxD[1][j]) for j in range(len(maxD[1]))])
mean_5u = np.mean(score_5u, axis = 0)
std_5u = np.std(score_5u, axis = 0)

"""Starting with 11 as initial, train on 11"""
score_11p = np.array([np.array(maxD[2][j]) for j in range(len(maxD[2]))])
mean_11p = np.mean(score_11p, axis = 0)
std_11p = np.std(score_11p, axis = 0)
score_11u = np.array([np.array(maxD[3][j]) for j in range(len(maxD[3]))])
mean_11u = np.mean(score_11u, axis = 0)
std_11u = np.std(score_11u, axis = 0)

"""Starting with 2 as initial, train on 31"""  # <-- Using this one!
score_31p = np.array([np.array(maxD[4][j]) for j in range(len(maxD[2]))])
mean_31p = np.mean(score_31p, axis = 0)
std_31p = np.std(score_31p, axis = 0)
score_31u = np.array([np.array(maxD[5][j]) for j in range(len(maxD[3]))])
mean_31u = np.mean(score_31u, axis = 0)
std_31u = np.std(score_31u, axis = 0)

# Create human-lambda database
score_31p = np.array([np.array(maxD[4][j]) for j in range(len(maxD[2]))])
print score_31p.shape
opt_sig = pd.DataFrame(score_31p.T, columns=['trial '+str(i+1) for i in range(20)])

opt_sig.index.name='time'
opt_sig['Play No.']=range(opt_sig.shape[0])
opt_sig = pd.melt(opt_sig, id_vars=['Play No.'], value_name='Best Score')
opt_sig['Kernel Type'] = 'Human'
print "Human Lam best: ", opt_sig['Best Score'].max()

# Create unit-lambda database
score_31u = np.array([np.array(maxD[5][j]) for j in range(len(maxD[3]))])
unit_sig = pd.DataFrame(score_31u.T, columns=['trial '+str(i+1) for i in range(20)])

unit_sig.index.name='time'
unit_sig['Play No.']=range(unit_sig.shape[0])
unit_sig = pd.melt(unit_sig, id_vars=['Play No.'], value_name='Best Score')
unit_sig['Kernel Type'] = 'Unit'
print "Unit Lam best: ", unit_sig['Best Score'].max()

# Get score history:
pre = Preprocess(all_dat='../all_games.pkl', pca_model='../eco_full_pca.pkl') # assuming local

# pre = Preprocess() # else, make local library
# pre.get_json('alluser_control.json')  # uncomment this to create the pkl file needed!!
# pre.train_pca()

X, y = pre.ready_player_one(2)

# Combine databases, add residual
sims = pd.concat([opt_sig, unit_sig])
sims['Residual'] = 43.2 - sims['Best Score']

#make test plot:

f,ax = plt.subplots(nrows = 2, figsize = (5,10))
sns.tsplot(sims, time='Play No.', unit='variable', value='Best Score', condition='Kernel Type',
           ci=[68], err_style=['ci_band'], ax=ax[0], estimator=np.mean)
sns.tsplot(sims, time='Play No.', unit='variable', value='Residual', condition='Kernel Type',
           ci=[68], err_style=['ci_band'], ax=ax[1], estimator=np.mean)
ax[1].set_yscale("log", nonposy='clip')
ax[1].set_ylim(.9, 100)
f.suptitle('Best Scoring Play over 30 trials', y=1.02)
y_res = np.array([np.max(y[:i]) for i in range(1, len(y)+1)])
ax[0].plot(y, 'r.')
ax[0].plot(y_res, 'k--')

ax[1].plot(43.2-y, 'r.')
ax[1].plot(43.2-y_res, 'k--')
# plt.text()
plt.tight_layout()
plt.draw()
print "showing test-plots: "
# Add best actual human plays over time:
roll_max = lambda y: np.array([np.max(y[:i]) for i in range(1,len(y)+1)])

mult=pd.pivot_table(pre.full_tab, index=['userid','id'])
mult.xs(1)
for i in range(1, 213):
    mult.loc[(slice(i, i), slice(None)), 'Play No.'] = range(len(mult.loc[(slice(i, i), slice(None)), :]))
    mult.loc[(slice(i, i), slice(None)), 'curr_best'] = roll_max(mult.loc[(slice(i, i), slice(None)), 'rem_nrg'].values)
mult.reset_index(inplace=True)  # remove multiindex

# Bootstrap confidence interval:
import scikits.bootstrap as boot
lo_hi = np.zeros((201, 2))
for i in range(201):
    lo_hi[i] = boot.ci(mult[mult['Play No.']==i]['curr_best'].values, np.mean, alpha=.32)
p_avg = mult[mult['Play No.']<=200.].groupby('Play No.')['curr_best'].mean()
p_std = mult[mult['Play No.']<=200.].groupby('Play No.')['curr_best'].std()

# Make the figure:
plt.figure(figsize=(10, 6))
sns.tsplot(sims, time='Play No.', unit='variable', value='Residual', condition='Kernel Type',
           ci=[68], err_style=['ci_band'], legend=False, estimator=np.mean)
plt.semilogy(43.64-y, 'r.', label='P2 data')
plt.plot(43.64-y_res, 'k--', label='P2 current best')

plt.plot(43.64-p_avg, 'k:', label='avg. best (all players)')
plt.fill_between(range(201), 43.64-lo_hi[:,0],43.64-lo_hi[:,1], alpha=.4, color='gray')

plt.title('Residual of Best Scoring Play over 30 trials\nby Kernel Type', y=1.02)
# plt.yscale('log')
plt.ylim(.9,100)
plt.text(205,1.3,'Human $\Lambda^*$')
plt.text(205,4,'Unit $\Lambda_I$')
plt.legend()
plt.tight_layout()
plt.show()
