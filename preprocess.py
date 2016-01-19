__author__ = 'Thurston Sexton'

from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD, FastICA
import numpy as np
import json
import pandas as pd
import os


class Preprocess:

    def __init__(self, pca_model=None, all_dat=None):

        if pca_model is not None:
            self.pca = joblib.load(pca_model)  # try 'eco_full_pca.pkl'

        self.full_tab = pd.read_json("../data.json")
        self.full_tab["rem_nrg"] = self.full_tab.apply(lambda x: self.remaining_energy(x.score), axis=1)

        if all_dat is not None:
            self.all_dat = joblib.load(all_dat)  # try 'all_games.pkl'
            drop = np.any(self.all_dat, axis=1)
            self.all_dat = self.all_dat[drop]
            self.full_tab = pd.read_json("../data.json")[drop]
            self.full_tab["rem_nrg"] = self.full_tab.apply(lambda x: self.remaining_energy(x.score), axis=1)
            self.proj = None
        # print os.system('pwd')



    @staticmethod
    def remaining_energy(consumption):
        max_batt = 0.55
        # consumption = np.linspace(0,2000000)
        # print consumption
        if consumption == -1:
            return 0
        else:
            return 100-(consumption/36000/max_batt)


    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except TypeError:
            return a

    def full_vec(self, pos, sig, size):
        series=np.zeros((size,), dtype=np.int)
        try:
            for i,x in enumerate(pos[:-1]):
                series[x:pos[i+1]] = sig[i]
        except Exception:
            pass
        #print series
        return series

    def get_json(self, file):

        with open(file) as json_data:
            data = json.load(json_data)

        self.dat=pd.DataFrame.from_dict(data['alluser_control'])
        self.dat["series"] = self.dat.apply(lambda x: self.totuple(self.full_vec(x['x'], x['sig'], 18160)),
                                  axis=1, raw=True)
        self.all_dat=np.empty((2391,18160))
        for i,x in enumerate(self.dat.x):
            self.all_dat[i,:]=self.full_vec(x, self.dat.sig[i], 18160)
        joblib.dump(self.all_dat, '../all_games.pkl')

    def train_pca(self, ndim=30):  # uses complete data-set
        # self.pca = TruncatedSVD(n_components=ndim)
        self.pca = FastICA(n_components=ndim)
        self.pca.fit(self.all_dat)
        joblib.dump(self.pca, '../eco_full_pca.pkl')  # save for later importing

    def ready_player_one(self, place):
        # place must be less than 7.
        top6 = [78, 122, 166, 70, 67, 69] #best players
        m1, m2, m3, m4, m5, m6 = [self.full_tab.userid.values==i for i in top6]
        masks = [m1, m2, m3, m4, m5, m6]
        X = self.all_dat[masks[place-1]]
        y = self.full_tab["rem_nrg"].values[masks[place-1]]

        X_pca = self.pca.transform(X)
        X_pca = np.vstack((X_pca.T, self.full_tab["finaldrive"].values[masks[place-1]])).T
        return (X_pca, y)

    def prep_by_id(self, play_no):
        id_no = self.full_tab['userid'][self.full_tab['id'] == play_no].values[0]
        # print id_no
        mask_a = self.full_tab.userid.values == id_no
        mask_b = self.full_tab.id.values <= play_no
        mask = np.logical_and(mask_a, mask_b)
        X = self.all_dat[mask]
        y = self.full_tab["rem_nrg"].values[mask]

        X_pca = self.pca.transform(X)
        X_pca = np.vstack((X_pca.T, self.full_tab["finaldrive"].values[mask])).T
        return (X_pca, y)