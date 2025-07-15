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
from Utils_Functions import KLdivergence
from sklearn.feature_selection import SelectKBest, SelectFpr,SelectPercentile 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from aif360.sklearn.metrics import equal_opportunity_difference,average_odds_difference
from Utils_Functions import generate_dataset, eod
import glob, copy
import re
from adf_utils.Measure import measure_final_score
#dataset ='Bank'
for dataset in  ['Adult' ,'Compas','Bank','Law','Heart','Student']:
    if dataset == 'Adult':
        sens_index = 7
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'adult_org-Copy1.csv'
        alg_list = ['ges','simy']
        
    if dataset == 'Compas':
        sens_index = 1
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'compas-Copy1'
        alg_list = ['ges','pc']
        
    if dataset == 'Bank':
        sens_index = 0
        priv_group = 5
        unpriv_group = 3
        data_file_name = 'bank'
        alg_list = ['ges']
        
    if dataset == 'Heart':
        sens_index = 0
        priv_group = 1
        unpriv_group = 0 
        data_file_name = 'heart_processed_1'
        alg_list = ['ges']
        
    if dataset == 'Law':
        sens_index = 1
        priv_group = 1
        unpriv_group = 0
        data_file_name = 'law.csv'
        alg_list = ['ges','simy']

    if dataset == 'Student':
        sens_index = 0
        priv_group = 1
        unpriv_group = 0 
        alg_list = ['simy','pc']
        data_file_name = 'students-processed_2'
        
    df = pd.read_csv('./subjects/datasets/'+data_file_name)
    df =  df.drop_duplicates().reset_index(drop=True)
    
    
    SelectKBest_final_EOD=[]
    SelectFpr_final_EOD=[]
    SelectPercentile_final_EOD=[]
    None_model_final_EOD = []
    SelectKBest_final_AOD=[]
    SelectFpr_final_AOD=[]
    SelectPercentile_final_AOD=[]
    None_model_final_AOD = []
    drop_final_EOD = []
    drop_final_AOD = []

    SelectKBest_EOD=[]
    SelectFpr_EOD=[]
    SelectPercentile_EOD=[]
    SelectKBest_AOD=[]
    SelectFpr_AOD=[]
    SelectPercentile_AOD=[]
    drop_EOD = []
    drop_AOD = []
    None_model = []    
    for round_ in range(10):
        
        X = df.to_numpy()[:,:-1]
        Y = df.to_numpy()[:,-1].astype(int)
        A = X[:,sens_index]
        
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(df[df.columns[:-1]],df[df.columns[-1]], df[df.columns[sens_index]], random_state=round_)
        
        #print(file_num,weights_ind)

        model_org = LogisticRegression()
        model_org.fit(X_train,y_train)

        preds_org = model_org.predict(X_test)
        acc_org = accuracy_score(y_test,preds_org)
        f1_org = f1_score(y_test,preds_org)
#         AOD_org = round(average_odds_difference(y_test.values, preds_org,prot_attr=A_test.values,priv_group=priv_group ),3)
#         EOD_org  = eod(y_test.values, preds_org,sens=A_test.values, priv=priv_group, unpriv=unpriv_group)
#         #print('Success rate DAG', file_num, succ_rate)
        yname = df.columns[-1]
        biased_col = df.columns[sens_index]
        X_test_df = copy.deepcopy(X_test)
        X_test_df[yname] = y_test
        EOD_org = measure_final_score(X_test_df, preds_org,biased_col , 'eod', yname,priv_group, unpriv_group)
        AOD_org = measure_final_score(X_test_df, preds_org,biased_col , 'aod', yname,priv_group, unpriv_group)
        for transformer in ['SelectKBest', 'SelectFpr','SelectPercentile' ,'drop']:


            if transformer == 'SelectKBest':
                Kbest = SelectKBest( k=int(X.shape[1]/2))
                X_new = Kbest.fit_transform(df[df.columns[:-1]], df[df.columns[-1]])
                X_new = pd.DataFrame(X_new, columns =Kbest.get_feature_names_out())
                
   
            elif transformer == 'SelectFpr':
                sfpr =  SelectFpr(alpha=0.01)
                X_new = sfpr.fit_transform(df[df.columns[:-1]], df[df.columns[-1]])
                X_new = pd.DataFrame(X_new, columns =sfpr.get_feature_names_out())
            elif transformer == 'SelectPercentile':
                percentile =  SelectPercentile(percentile=40)
                
                X_new = percentile.fit_transform(df[df.columns[:-1]], df[df.columns[-1]])
                X_new = pd.DataFrame(X_new, columns =percentile.get_feature_names_out())
            elif transformer == 'drop':  
                
                X_new =  df[df.columns[:-1]].drop(columns = [biased_col])
#             print(transformer,X_new.shape[1],X_test.shape[1])    
            X_train_new, X_test_new = train_test_split(X_new, random_state=round_)
            
            
            
            
            model = LogisticRegression()
            model.fit(X_train_new,y_train)
            preds_new = model.predict(X_test_new)
            acc_new = accuracy_score(y_test,preds_new)
            f1_new  = f1_score(y_test,preds_new)
            
            EOD_new = measure_final_score(X_test_df, preds_new,biased_col , 'eod', yname,priv_group, unpriv_group)
            AOD_new = measure_final_score(X_test_df, preds_new,biased_col , 'aod', yname,priv_group, unpriv_group) 
            
            
#             AOD_new  = round(average_odds_difference(y_test, preds_new,prot_attr=A_test,priv_group=priv_group ),3)
#             EOD_new  = eod(y_test, preds_new,sens=A_test, priv=priv_group, unpriv=unpriv_group)
            acc_diff = round(acc_new - acc_org,2)
            f1_diff = round(f1_new- f1_org,2)
            EOD_diff = EOD_new - EOD_org
            AOD_diff = AOD_new - AOD_org
            EOD_diff_abs = abs(EOD_new) - abs(EOD_org)
            AOD_diff_abs = abs(AOD_new) - abs(AOD_org)

            if transformer == 'SelectKBest':
                SelectKBest_EOD.append([ EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                SelectKBest_AOD.append([ AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

            elif transformer == 'SelectFpr':

                SelectFpr_EOD.append([EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                SelectFpr_AOD.append([AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

            elif transformer == 'SelectPercentile':
                SelectPercentile_EOD.append([EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                SelectPercentile_AOD.append([AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

            elif transformer == 'drop':
                drop_EOD.append([EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                drop_AOD.append([AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

    
                
                
           
    SelectKBest_EOD = np.round(np.mean(SelectKBest_EOD,axis=0),2)
    SelectFpr_EOD = np.round(np.mean(SelectFpr_EOD,axis=0),2)
    SelectPercentile_EOD = np.round(np.mean(SelectPercentile_EOD,axis=0),2)
    drop_EOD = np.round(np.mean(drop_EOD,axis=0),2)
#         print(np.array(SelectKBest_final_EOD).min(),np.array(SelectKBest_final_EOD).max())
#         print(np.array(SelectKBest_final_AOD).min(),np.array(SelectKBest_final_AOD).max())

#         print(np.array(SelectFpr_final_EOD).min(),np.array(SelectFpr_final_EOD).max())
#         print(np.array(SelectFpr_final_AOD).min(),np.array(SelectFpr_final_AOD).max())

#         print(np.array(SelectPercentile_final_EOD).min(),np.array(SelectPercentile_final_EOD).max())
#         print(np.array(SelectPercentile_final_AOD).min(),np.array(SelectPercentile_final_AOD).max())
# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_EOD_' + str(file_num)+'.npy',SelectKBest_final_EOD)
# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_AOD_' + str(file_num)+'.npy',SelectKBest_final_AOD)

# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_EOD_' + str(file_num)+'.npy',SelectFpr_final_EOD)
# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_AOD_' + str(file_num)+'.npy',SelectFpr_final_AOD)

# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_EOD_' + str(file_num)+'.npy',SelectPercentile_final_EOD)
# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_AOD_' + str(file_num)+'.npy',SelectPercentile_final_AOD)

# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_EOD_' + str(file_num)+'.npy',drop_final_EOD)
# np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_AOD_' + str(file_num)+'.npy',drop_final_AOD)

    print(f' {dataset}  {SelectKBest_EOD[0]} & {SelectKBest_EOD[1]} & {SelectKBest_EOD[2]} & {SelectFpr_EOD[0]} & {SelectFpr_EOD[1]} & {SelectFpr_EOD[2]} & {SelectPercentile_EOD[0]} & {SelectPercentile_EOD[1]} & {SelectPercentile_EOD[2]} & {drop_EOD[0]} & {drop_EOD[1]} & {drop_EOD[2]}')