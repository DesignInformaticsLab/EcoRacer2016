import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import subprocess


method_string = ['original_learned_sigma_from_all_plays_from_all_player2_03212016_', # all plays, all sigma
                 'original_learned_sigma_from_5_plays_from_5_player2_03252016_', # 5 plays, 5 sigma
                 'original_learned_sigma_from_11_plays_from_11_player2_03252016_)', # 11 plays, 11 sigma
                 'original_unit_sigma_from_11_player2_03252016_', # 11 plays, unit sigma
                 'original_unit_sigma_from_5_player2_03252016_', # 5 plays, unit sigma
                 'original_unit_sigma_from_31_player2_03252016_'
                ]
iter = 20

for method in method_string:
    for i in range(iter):
        d = read_db(method+str(i))


def read_db(method):
    conn = psycopg2.connect("dbname=postgres user=postgres password=GWC464doi")
    cur = conn.cursor()
    cur.execute("SELECT * from ecoracer_learning_ego_table where method=" + method + ";")
    psdb = cur.fetchall()
    cur.close()
    print("read initial database done!")
    return psdb