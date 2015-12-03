# to run the optimization in matlab

__author__ = 'p2admin'
import sys
from ego import Kriging
from preprocess import Preprocess
import numpy as np


pre = Preprocess(pca_model='eco_full_pca.pkl', all_dat='all_games.pkl')
X, y = pre.ready_player_one(2)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)
soln = CovarianceEstimate(X, y)

if __name__ == '__main__':
    x = float(sys.argv[1])
    return soln.model.obj(x0)