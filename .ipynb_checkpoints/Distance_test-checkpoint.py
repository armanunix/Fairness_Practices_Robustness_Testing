import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("./subjects/")
import pickle
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
import tensorflow as tf
import time
import random, math
import tensorflow_probability as tfb
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import  euclidean_distances
import glob
import re
from sklearn.neighbors import KDTree
from Utils_Functions import generate_dataset, eod
from sklearn.model_selection import train_test_split
dataset ='Adult'
for dataset in ['Bank']:#,'Compas','Bank','Heart','Law','Student']:
    if dataset == 'Adult':
        sens_index = 7
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'adult_org-Copy1.csv'
    if dataset == 'Compas':
        sens_index = 1
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'compas-Copy1'
    if dataset == 'Bank':
        sens_index = 0
        priv_group = 5
        unpriv_group = 3
        data_file_name = 'bank'
    if dataset == 'Heart':
        sens_index = 0
        priv_group = 1
        unpriv_group = 0 
        data_file_name = 'heart_processed_1'
    if dataset == 'Law':
        sens_index = 1
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'law.csv'

    if dataset == 'Student':
        sens_index = 0
        priv_group = 1
        unpriv_group = 0  
        data_file_name = 'students-processed_2'
    df = pd.read_csv('./subjects/datasets/'+data_file_name)
    df = df.drop_duplicates()
    
    
    
    
    X = df.to_numpy()[:,:-1]
    Y1 = df.to_numpy()[:,-1]
    acc_test =[]
    acc_test_rnd = []
    for i in range(10):
        X1, X_test = train_test_split(X)

        num_cluster = 100
        KMean = KMeans(n_clusters=num_cluster)
        KMean.fit(X1)

        if dataset=='Adult':
            cont_cols=['hr']
        elif dataset=='Law':
            cont_cols=['UGPA']

        rnd_df = pd.DataFrame(columns=df.columns)

        for col in rnd_df.columns:
            if col in cont_cols:
                rnd_df[col] = np.random.uniform(df[col].min(),df[col].max(),X_test.shape[0])
            else:
                rnd_df[col] = np.random.randint(df[col].min(),df[col].max(),X_test.shape[0])
        ave_dist =[] 
        for i in range(KMean.n_clusters):
            mean_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).mean()
            std_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).std()
            if dataset == 'Heart':
                ave_dist.append(mean_dist+ (3 * std_dist))
            elif dataset == 'Compas':
                ave_dist.append(mean_dist+ (1 * std_dist))
            elif dataset == 'Student':
                ave_dist.append(mean_dist+ (1 * std_dist))
            else:
                ave_dist.append(mean_dist)
        final_res=[]
        for i in range(X_test.shape[0]):
            if euclidean_distances([X_test[i]],KMean.cluster_centers_).min() <= max(ave_dist):
                final_res.append(1)
            else :
                final_res.append(0)
        final_res = np.array(final_res)
        acc_test.append(np.where(np.array(final_res)==1)[0].shape[0]/final_res.shape[0])
        print(dataset, acc_test[-1])   

        X_test_rnd = rnd_df.to_numpy()[:,:-1]
        final_res_rnd=[]
        for i in range(X_test_rnd.shape[0]):
            if euclidean_distances([X_test_rnd[i]],KMean.cluster_centers_).min() <= max(ave_dist):
                final_res_rnd.append(1)
            else :
                final_res_rnd.append(0)
        final_res_rnd = np.array(final_res_rnd)
        acc_test_rnd.append(np.where(np.array(final_res_rnd)==1)[0].shape[0]/final_res_rnd.shape[0])
        print(dataset+' RND', acc_test_rnd[-1])
    print('Acc test',np.mean(acc_test) )
    print('Acc rnd test',np.mean(acc_test_rnd) )