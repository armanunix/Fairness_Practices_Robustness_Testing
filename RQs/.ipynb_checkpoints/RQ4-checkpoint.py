import warnings
warnings.filterwarnings('ignore')
import sys,os
sys.path.append("./subjects/")
subject_path = os.path.join(os.getcwd(), '..', 'subjects')
sys.path.append(os.path.abspath(subject_path))
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
import glob
import re
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds
import argparse
def generate_dataset_shift(data, graph, edges, sens_index,priv_group, unpriv_group):

    dataset_types = [str(data[i].dtype) for i in data.columns]
    succ_generated = 0
    generation_coef = 1
    graph_dic ={}
    for i in graph.columns[1:]:
        if np.where(graph[i])[0].shape[0]==0:
            graph_dic[i]=None
        else:
            graph_dic[i]= graph['Unnamed: 0'][np.where(graph[i])[0]].values

    final_df = pd.DataFrame(columns = data.columns) 
    trial = 0
    not_interesting = False
    while final_df.shape[0]<data.shape[0]:

        if trial > 20:
            not_interesting = True
            return None , False
        df_new_dic ={}
        for edge in graph.sum().index[np.where(graph.sum()==0)[0]]:
            df_new_dic[edge] = np.random.choice(np.unique(data[edge]), size = data.shape[0]*generation_coef) 

        statring_atts = graph.sum().index[np.where(graph.sum()==0)[0]]
        
        while statring_atts.shape[0] != graph['Unnamed: 0'].shape[0]:
            
            for att in graph_dic.keys():
                if att not in statring_atts:
                    if 0 in  [1 if graph_dic[att][i] in statring_atts else 0 for i in range(graph_dic[att].shape[0])]:

                        continue
                    else:
                        edge_logits = 0
                        
                        for cause in graph_dic[att]:
                            edge_logits += (edges[cause+att] * df_new_dic[cause])
                        if np.unique(data[att]).shape[0]==2:
                            df_new_dic[att] =  tfb.distributions.Bernoulli(logits=edge_logits + edges[att+'0'] ).sample().numpy()
                        elif 'float' in dataset_types[np.where(data.columns==att)[0][0]]:

                            df_new_dic[att] =  tfb.distributions.Normal(loc=(edge_logits+ edges[att+'0']), scale= edges['sigma_h']).sample().numpy()
                        
                        else:    
                            df_new_dic[att] =  tfb.distributions.Poisson(rate=tf.exp(edge_logits+ edges[att+'0']) ).sample().numpy()  

                        statring_atts = np.append(statring_atts,att) 

        new_df = pd.DataFrame(columns = data.columns)
        for col in new_df.columns:
            new_df[col] = df_new_dic[col]
        
        ind_inf = np.unique(np.where(new_df>data.max())[0])
        new_df.drop(ind_inf,axis=0,inplace=True)
        for col in range(new_df.columns.shape[0]):
            new_df[new_df.columns[col]]=new_df[new_df.columns[col]].astype(dataset_types[col])
        if new_df.shape[0]<1:
            return None , False
        X2 = new_df.to_numpy()[:,:-1]
        Y2 = new_df.to_numpy()[:,-1]
        #dist = euclidean_distances(X2, centroids)
        #succ_generated += new_df.iloc[np.where((ave_dist>=dist).sum(1)>0)].shape[0]
        final_df = pd.concat([final_df,new_df]).reset_index(drop=True)
        final_df = final_df.drop_duplicates()


            
        trial += 1
   
    #final_df = final_df.astype(int)
    Y2 = final_df.to_numpy()[:,-1]
    if (Y2.sum()/Y2.shape[0]< 0.06) or (Y2.sum()/Y2.shape[0]> 0.95):
        return None,  False
    if priv_group not in final_df[final_df.columns[sens_index]].values or unpriv_group not in final_df[final_df.columns[sens_index]].values:
        return None,  False


#dataset ='Bank'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Select the dataset')
parser.add_argument('--mitigator', type=str, help='TO or CEO')
args = parser.parse_args()
dataset=args.dataset
Practice=args.mitigator
for dataset in [dataset]:
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
        
    df = pd.read_csv('../subjects/datasets/'+data_file_name)
    df =  df.drop_duplicates().reset_index(drop=True)

    X1 = df.to_numpy()[:,:-1]
    Y1 = df.to_numpy()[:,-1].astype(int)

    
    for Algorithm in alg_list:
        print('Algorithm', Algorithm)
        for edge_list_filename in glob.glob('../'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
            print(edge_list_filename)
            file_num = int(re.findall(r'\d+', edge_list_filename.split('/')[-1])[0])
            RQ1_res = np.load('../'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy')
            if RQ1_res[np.where(RQ1_res[:,1].astype(int)==file_num)[0]][0][2].astype(float)==0.0:
                print('No',file_num)
                continue
            try:
                graph_filename = '../'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
                graph = pd.read_csv(graph_filename)
                #edge_list_filename = './'+dataset+'_Analysis/'+Algorithm+'/PP/'+dataset+'_'+Algorithm+'_pp_{file_num}.csv'.format(file_num=file_num)
                edges_list = pd.read_csv(edge_list_filename)
                if 'first_pf0' not in edges_list.columns and 'label0' not in edges_list.columns and 'G30' not in edges_list.columns and 'y0' not in edges_list.columns:
                    continue 
                if dataset=='Bank' and Algorithm=='simy':
                    graph.columns = [i.replace('1','') for i in graph.columns]
                    graph[graph.columns[0]] = [i.replace('1','') for i in graph[graph.columns[0]]]
                    edges_list.columns = [i.replace('1','') for i in edges_list.columns]
                edges_list = edges_list[edges_list.columns[1:-1]]

            except:
                print('Not a DAG! ',file_num)
                continue

            X1_coef = edges_list.to_numpy()
            KMean_coef = KMeans(n_clusters=10)
            KMean_coef.fit(X1_coef)
            
            mitigator_final_EOD=[]
            None_model_final_EOD = []
            mitigator_final_AOD=[]
            None_model_final_AOD = []

            for i in range(KMean_coef.n_clusters):
                #print('Coef ',i)
                weights_ind = np.random.choice(np.where(KMean_coef.labels_==i)[0])
                edges = edges_list.iloc[weights_ind]

                if dataset=='Law':
                    edges['first_pf0']+=1
                elif dataset=='Bank':
                    edges['label0']+=1
                elif dataset=='Heart':
                    edges['label0']+=1
                elif dataset=='Student':
                    edges['G30']+=1
                else:
                    edges['y0']+=1
                #print(file_num,weights_ind)
                mitigator_EOD=[]
                mitigator_AOD=[]
                None_model = []
                for j in range(10):
                    final_df, status = generate_dataset_shift(df, graph, edges,sens_index, priv_group, unpriv_group)

                    if status==False:
                        continue

                    X2 = final_df.to_numpy()[:,:-1]
                    Y2 = final_df.to_numpy()[:,-1].astype(int)
                    A2 = X2[:,sens_index]

                    if priv_group not in A2 or unpriv_group not in A2:
                        print('No sens group')
                        continue
                    model = LogisticRegression()
                    model.fit(X2,Y2)
                    
                    preds_None = model.predict(X2)
                    acc_None = accuracy_score(Y2,preds_None)
                    f1_None = f1_score(Y2,preds_None)
                    AOD_None = round(average_odds_difference(Y2, preds_None,prot_attr=A2,priv_group=priv_group ),3)
                    EOD_None = eod(Y2, preds_None,sens=A2, priv=priv_group, unpriv=unpriv_group)        
                    None_model.append([EOD_None,acc_None,f1_None])
                    y_prob = model.predict_proba(X2)
                    if Practice == 'TO':
                    #print('Success rate DAG', file_num, succ_rate)
                        mitigator = ThresholdOptimizer(
                            estimator=model,
                            constraints="equalized_odds",   # or "demographic_parity"
                            prefit=True,
                            predict_method="predict_proba")
                        mitigator.fit(X2,Y2, sensitive_features=A2)
                        preds_new = mitigator.predict(X2,sensitive_features=A2)
                    if Practice == 'CEO':
                        mitigator = CalibratedEqualizedOdds(prot_attr=A2, cost_constraint='fpr')
                        mitigator.fit(y_prob,Y2)
                        preds_new = mitigator.predict(y_prob)
                    acc_temp = accuracy_score(Y2,preds_new)
                    f1_temp = f1_score(Y2,preds_new)
                    AOD = round(average_odds_difference(Y2, preds_new,prot_attr=A2,priv_group=priv_group ),3)
                    EOD = eod(Y2, preds_new,sens=A2, priv=priv_group, unpriv=unpriv_group)
                    acc_diff = round(acc_temp - acc_None,2)
                    f1_diff = round(f1_temp- f1_None,2)
                    EOD_diff = EOD - EOD_None
                    AOD_diff = AOD - AOD_None
                    EOD_diff_abs = abs(EOD) - abs(EOD_None)
                    AOD_diff_abs = abs(AOD) - abs(AOD_None)

                    input(EOD_diff)
                    
                    mitigator_EOD.append([ EOD,EOD_None,EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                    mitigator_AOD.append([ AOD,AOD_None,AOD_diff,AOD_diff_abs, acc_diff, f1_diff])



                mitigator_final_EOD.append(np.mean(mitigator_EOD,axis=0))
                mitigator_final_AOD.append(np.mean(mitigator_AOD,axis=0)) 
                
            np.save('../'+dataset+'_Analysis/RQ4/'+Algorithm+'_'+Practice +'_'+ str(file_num)+'.npy',mitigator_final_EOD)
            np.save('../'+dataset+'_Analysis/RQ4/'+Algorithm+'_'+Practice +'_' + str(file_num)+'.npy',mitigator_final_AOD)


