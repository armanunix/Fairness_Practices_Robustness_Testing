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
from subjects.Utils_Functions import generate_dataset, eod
import mahalanobis
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
dataset ='Bank'

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

X1 = df.to_numpy()[:,:-1]
Y1 = df.to_numpy()[:,-1]
mahND = mahalanobis.MahalanobisND(X1,10 )
num_cluster = 100
try :
    with open('./'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster), 'rb') as f:
        KMean = pickle.load(f)
except:
    KMean = KMeans(n_clusters=num_cluster)
    KMean.fit(X1)
    with open('./'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster),'wb') as f:
        pickle.dump(KMean,f)

ave_dist =[] 
for i in range(KMean.n_clusters):
    mean_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).mean()
    std_dist = euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).std()
    if dataset == 'Heart':
        ave_dist.append(mean_dist+ (3 * std_dist))
    elif dataset == 'Compas':
        ave_dist.append(mean_dist+ (2 * std_dist))
    elif dataset == 'Student':
        ave_dist.append(mean_dist+ (3 * std_dist))
    elif dataset == 'Bank':
        ave_dist.append(mean_dist+ (3 * std_dist))
    else:
        ave_dist.append(mean_dist+ (2 * std_dist))
    
for Algorithm in ['pc','ges','simy']:
    print('Algorithm', Algorithm)
    succ_rate_list =[]
    metrics_temp=[]
    for edge_list_filename in glob.glob('./'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
#         print(edge_list_filename)
        file_num = int(re.findall(r'\d+', edge_list_filename)[0])
        
    #     if file_num in [5,7,10,12]:
    #         continue

        try:
            graph_filename = './'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
            graph = pd.read_csv(graph_filename)
            

            #edge_list_filename = './'+dataset+'_Analysis/'+Algorithm+'/PP/'+dataset+'_'+Algorithm+'_pp_{file_num}.csv'.format(file_num=file_num)
            edges_list = pd.read_csv(edge_list_filename)

            if dataset=='Bank' and Algorithm=='simy':
                graph.columns = [i.replace('1','') for i in graph.columns]
                graph[graph.columns[0]] = [i.replace('1','') for i in graph[graph.columns[0]]]
                edges_list.columns = [i.replace('1','') for i in edges_list.columns]
            edges_list = edges_list[edges_list.columns[1:-1]].mean()

        except:
            print('Not a DAG! ',file_num)
            continue
        

  
            #ave_dist.append(euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).max())
        

        final_df, succ_rate = generate_dataset(df, graph, edges_list, ave_dist, KMean.cluster_centers_ ,sens_index, priv_group, unpriv_group)


        if succ_rate == 0.0:
            continue


#             metrics_temp.append(np.array([Algorithm,file_num,round(succ_rate,3),distance_avg]))
#             print('Success rate DAG ', file_num,'-> Succ = ', round(succ_rate,3),' Dis avg= ', distance_avg,' Dis std= ', distance_std)
        else:
            X_org = df.to_numpy()[:,:-1]
            X_gen = final_df.to_numpy()[:,:-1]
            tree = KDTree(X_org)
            distance_list = tree.query(X_gen, k=1)[0]
            distance_avg = round(distance_list.mean(),3)
            mah_dist  = mahND.calc_distances(final_df.to_numpy().astype(float)[:,:-1]).mean()
        succ_rate_list.append(np.array([Algorithm,file_num,succ_rate,distance_avg, mah_dist]))
    
    print(f'{dataset}-{Algorithm} -> ${round(np.mean(np.array(succ_rate_list)[:,2:].astype(float), axis=0)[0],2)}$ & ${round(np.std(np.array(succ_rate_list)[:,2:].astype(float), axis=0)[0],2)}$ & ${round(np.min(np.array(succ_rate_list)[:,2:].astype(float), axis=0)[0],2)}$ & ${round(np.max(np.array(succ_rate_list)[:,2:].astype(float), axis=0)[0],2)}$ & ${round(np.mean(np.array(succ_rate_list)[:,2:].astype(float), axis=0)[1],1)}$&${round(np.mean(np.array(succ_rate_list)[:,-1].astype(float), axis=0),1)}$')
    np.save('./'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy',metrics_temp)