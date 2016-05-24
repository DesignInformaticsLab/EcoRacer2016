import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import subprocess
import seaborn as sns
import pandas as pd
import os.path

# full_method_string = ['original_learned_sigma_from_all_plays_from_all_player2_03212016_', # all plays, all sigma
#                  'original_learned_sigma_from_5_plays_from_5_player2_03252016_', # 5 plays, 5 sigma
#                  'original_learned_sigma_from_11_plays_from_11_player2_03252016_', # 11 plays, 11 sigma
#                  'original_unit_sigma_from_11_player2_03252016_', # 11 plays, unit sigma
#                  'original_unit_sigma_from_5_player2_03252016_', # 5 plays, unit sigma
#                 ]

# use the following to address identifier truncation issue
method_string = ['5_plays_from_5_player2_04042016_', # 5 plays, 5 sigma
                 'original_unit_sigma_from_5_player2_03252016_', # 5 plays, unit sigma
                 '11_plays_from_11_player2_04042016_', # 11 plays, 11 sigma
                 'original_unit_sigma_from_11_player2_03252016_', # 11 plays, unit sigma
                 '31_plays_from_2_player2_04212016_', # 2 plays, 31 sigma
                 'original_unit_sigma_from_2_player2_04212016_', # 2 plays, unit sigma
                ]


iter = 20
data = [[[] for i in range(iter)] for j in range(len(method_string))]
max = [[[] for i in range(iter)] for j in range(len(method_string))]

data_address = 'simulation_04042016'
if os.path.isfile(data_address+'.npy'):
    max = np.load(data_address+'.npy')
else:
    for j, method in enumerate(method_string):
        for i in range(iter):
            conn = psycopg2.connect("dbname=postgres user=postgres password=GWC464doi")
            cur = conn.cursor()
            cur.execute("SELECT iteration, score FROM ecoracer_learning_ego_table WHERE method LIKE '%" + method+str(i+1) + "';")
            psdb = cur.fetchall()
            cur.close()
            data[j][i] = psdb
            print("read initial database done!")

            plays = np.array(psdb)
            score = plays[:,1]
            max_score = [np.max(score[0:ii]) for ii in range(1,score.size)]
            max[j][i] = max_score
    np.save(data_address, max)

score_5p = np.array([np.array(max[0][j]) for j in range(len(max[0]))])
mean_5p = np.mean(score_5p, axis = 0)
std_5p = np.std(score_5p, axis = 0)
score_5u = np.array([np.array(max[1][j]) for j in range(len(max[1]))])
mean_5u = np.mean(score_5u, axis = 0)
std_5u = np.std(score_5u, axis = 0)

score_11p = np.array([np.array(max[2][j]) for j in range(len(max[2]))])
mean_11p = np.mean(score_11p, axis = 0)
std_11p = np.std(score_11p, axis = 0)
score_11u = np.array([np.array(max[3][j]) for j in range(len(max[3]))])
mean_11u = np.mean(score_11u, axis = 0)
std_11u = np.std(score_11u, axis = 0)

score_31p = np.array([np.array(max[4][j]) for j in range(len(max[2]))])
mean_31p = np.mean(score_31p, axis = 0)
std_31p = np.std(score_31p, axis = 0)
score_31u = np.array([np.array(max[5][j]) for j in range(len(max[3]))])
mean_31u = np.mean(score_31u, axis = 0)
std_31u = np.std(score_31u, axis = 0)

plt.subplot(3,1,1)
plt.errorbar(range(mean_5p.size), mean_5p, yerr=std_5p)
plt.errorbar(range(mean_5u.size), mean_5u, yerr=std_5u)
plt.title('5 plays as initial | sigma from 5 plays vs unit sigma')

plt.subplot(3,1,2)
plt.errorbar(range(mean_11p.size), mean_11p, yerr=std_11p)
plt.errorbar(range(mean_11u.size), mean_11u, yerr=std_11u)
plt.title('11 plays as initial | sigma from 11 plays vs unit sigma')

plt.subplot(3,1,3)
plt.errorbar(range(mean_31p.size), mean_31p, yerr=std_31p)
plt.errorbar(range(mean_31u.size), mean_31u, yerr=std_31u)
plt.title('2 plays as initial | sigma from 31 plays vs unit sigma')

plt.show()


