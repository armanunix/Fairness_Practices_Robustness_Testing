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
import glob
import re
import argparse
    
def generate_dataset_I(data, graph, edges, ave_dist, centroids,sens_index, priv_group, unpriv_group):

    dataset_types = [str(data[i].dtype) for i in data.columns]
    succ_generated = 0
    generation_coef = 10
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
            return None , 0.0#succ_generated/( trial *  data.shape[0]*generation_coef) 
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
            return None , 0.0
        X2 = new_df.to_numpy()[:,:-1]
        Y2 = new_df.to_numpy()[:,-1]
        dist = euclidean_distances(X2, centroids)
        succ_generated += new_df.iloc[np.where((ave_dist>=dist).sum(1)>0)].shape[0]
        final_df = pd.concat([final_df,new_df.iloc[np.where((ave_dist>=dist).sum(1)>0)[0]]]).reset_index(drop=True)
        final_df = final_df.drop_duplicates()

        #print(succ_generated,trial)
        if succ_generated<10:
            return None, 0.0
            
        trial += 1
   
    #final_df = final_df.astype(int)
    succ_rate = succ_generated/( trial *  data.shape[0]*generation_coef) 
    final_df = final_df.sample(n= data.shape[0])
    Y2 = final_df.to_numpy()[:,-1]
    if (Y2.sum()/Y2.shape[0]< 0.06) or (Y2.sum()/Y2.shape[0]> 0.95):
        return None, 0.0#succ_generated/( trial *  data.shape[0]*generation_coef)   
    if priv_group not in final_df[final_df.columns[sens_index]].values or unpriv_group not in final_df[final_df.columns[sens_index]].values:
        return None, 0.0

    return final_df, succ_rate

def eod(y_true, y_pred, sens, priv, unpriv):
    
    ind_priv = np.where(sens==priv)[0]
    ytru_rate_priv = np.where(y_true[ind_priv]==1)[0].shape[0]
    pred_rate_priv = np.where(y_pred[ind_priv]==1)[0].shape[0]
    tpr_priv = ytru_rate_priv/ind_priv.shape[0] - pred_rate_priv/ind_priv.shape[0]
    ind_unpriv = np.where(sens==unpriv)[0]
    ytru_rate_unpriv = np.where(y_true[ind_unpriv]==1)[0].shape[0]
    pred_rate_unpriv = np.where(y_pred[ind_unpriv]==1)[0].shape[0]
    tpr_unpriv = ytru_rate_unpriv/ind_unpriv.shape[0] - pred_rate_unpriv/ind_unpriv.shape[0]
    return round(tpr_priv - tpr_unpriv , 3)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Select the dataset')
args = parser.parse_args()
dataset=args.dataset

for dataset in [dataset]:
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
    df = pd.read_csv('../subjects/datasets/'+data_file_name)
    df = df.drop_duplicates()

    X1 = df.to_numpy()[:,:-1]
    Y1 = df.to_numpy()[:,-1].astype(int)
    num_cluster = 100
    try :
        with open('../'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster), 'rb') as f:
            KMean = pickle.load(f)
    except:
        KMean = KMeans(n_clusters=num_cluster)
        KMean.fit(X1)
        with open('../'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster),'wb') as f:
            pickle.dump(KMean,f)

    ave_dist =[] 
    for i in range(KMean.n_clusters):
        mean_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).mean()
        std_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).std()
        ave_dist.append(mean_dist + (2 * std_dist))
        #ave_dist.append(euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).max())

    for Algorithm in ['pc','ges','simy']:
        print('Algorithm', Algorithm)
        for edge_list_filename in glob.glob('../'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
            print(edge_list_filename)
            file_num = int(re.findall(r'\d+', edge_list_filename)[0])
            RQ1_res = np.load('../'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy')
            if RQ1_res[np.where(RQ1_res[:,1].astype(int)==file_num)[0]][0][2].astype(float)==0.0:
                continue
            try:
                graph_filename = '../'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
                graph = pd.read_csv(graph_filename)
                #edge_list_filename = './'+dataset+'_Analysis/'+Algorithm+'/PP/'+dataset+'_'+Algorithm+'_pp_{file_num}.csv'.format(file_num=file_num)
                edges_list = pd.read_csv(edge_list_filename)
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

            for i in range(KMean_coef.n_clusters):
                #print('Coef ',i)
                edges = edges_list.iloc[np.random.choice(np.where(KMean_coef.labels_==i)[0])]
                SelectKBest_EOD=[]
                SelectFpr_EOD=[]
                SelectPercentile_EOD=[]
                SelectKBest_AOD=[]
                SelectFpr_AOD=[]
                SelectPercentile_AOD=[]
                drop_EOD = []
                drop_AOD = []
                None_model = []
                for j in range(10):
                    final_df, succ_rate = generate_dataset_I(df, graph, edges, ave_dist, KMean.cluster_centers_ ,sens_index, priv_group, unpriv_group)
                    if succ_rate==0.0 :
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
                    

                    #print('Success rate DAG', file_num, succ_rate)
                    for transformer in ['SelectKBest', 'SelectFpr','SelectPercentile' ,'drop']:


                        if transformer == 'SelectKBest':
                            Kbest = SelectKBest( k=int(X2.shape[1]/2))
                            X_new = Kbest.fit_transform(X2, Y2)
                        elif transformer == 'SelectFpr':
                            sfpr =  SelectFpr(alpha=0.01)
                            X_new = sfpr.fit_transform(X2, Y2)

                        elif transformer == 'SelectPercentile':
                            percentile =  SelectPercentile(percentile=10)
                            X_new = percentile.fit_transform(X2, Y2)
                        elif transformer == 'drop':                        
                            X_new =  np.delete(X2,sens_index,axis=1)


                        model = LogisticRegression()
                        model.fit(X_new,Y2)
                        preds_new = model.predict(X_new)
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
                        #print([ EOD_diff, acc_diff, f1_diff])
                        if transformer == 'SelectKBest':
                            SelectKBest_EOD.append([ EOD,EOD_None,EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                            SelectKBest_AOD.append([ AOD,AOD_None,AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

                        elif transformer == 'SelectFpr':
                            SelectFpr_EOD.append([EOD,EOD_None,EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                            SelectFpr_AOD.append([AOD,AOD_None,AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

                        elif transformer == 'SelectPercentile':
                            SelectPercentile_EOD.append([EOD,EOD_None,EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                            SelectPercentile_AOD.append([AOD,AOD_None,AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

                        elif transformer == 'drop':
                            drop_EOD.append([EOD,EOD_None,EOD_diff,EOD_diff_abs, acc_diff, f1_diff])
                            drop_AOD.append([AOD,AOD_None,AOD_diff,AOD_diff_abs, acc_diff, f1_diff])

                if len(SelectKBest_EOD)>0:
                    SelectKBest_final_EOD.append(np.mean(SelectKBest_EOD,axis=0))
                if len(SelectKBest_AOD)>0:
                    SelectKBest_final_AOD.append(np.mean(SelectKBest_AOD,axis=0)) 

                if len(SelectFpr_EOD)>0:
                    SelectFpr_final_EOD.append(np.mean(SelectFpr_EOD,axis=0))
                if len(SelectFpr_AOD)>0:
                    SelectFpr_final_AOD.append(np.mean(SelectFpr_AOD,axis=0))

                if len(SelectPercentile_EOD)>0:    
                    SelectPercentile_final_EOD.append(np.mean(SelectPercentile_EOD,axis=0))
                if len(SelectPercentile_AOD)>0:    
                    SelectPercentile_final_AOD.append(np.mean(SelectPercentile_AOD,axis=0))

                if len(drop_EOD)>0:    
                    drop_final_EOD.append(np.mean(drop_EOD,axis=0))
                if len(drop_AOD)>0:    
                    drop_final_AOD.append(np.mean(drop_AOD,axis=0))

    #         print(np.array(SelectKBest_final_EOD).min(),np.array(SelectKBest_final_EOD).max())
    #         print(np.array(SelectKBest_final_AOD).min(),np.array(SelectKBest_final_AOD).max())

    #         print(np.array(SelectFpr_final_EOD).min(),np.array(SelectFpr_final_EOD).max())
    #         print(np.array(SelectFpr_final_AOD).min(),np.array(SelectFpr_final_AOD).max())

    #         print(np.array(SelectPercentile_final_EOD).min(),np.array(SelectPercentile_final_EOD).max())
    #         print(np.array(SelectPercentile_final_AOD).min(),np.array(SelectPercentile_final_AOD).max())
            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_EOD_' + str(file_num)+'.npy',SelectKBest_final_EOD)
            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_AOD_' + str(file_num)+'.npy',SelectKBest_final_AOD)

            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_EOD_' + str(file_num)+'.npy',SelectFpr_final_EOD)
            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_AOD_' + str(file_num)+'.npy',SelectFpr_final_AOD)

            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_EOD_' + str(file_num)+'.npy',SelectPercentile_final_EOD)
            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_AOD_' + str(file_num)+'.npy',SelectPercentile_final_AOD)

            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_EOD_' + str(file_num)+'.npy',drop_final_EOD)
            np.save('../'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_AOD_' + str(file_num)+'.npy',drop_final_AOD)

