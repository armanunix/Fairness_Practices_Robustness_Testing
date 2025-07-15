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
import mahalanobis
def generate_dataset_RND(data, graph, edges, ave_dist, centroids,sens_index, priv_group, unpriv_group):
    global final_df,dist
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
    while final_df.shape[0]<data.shape[0]/2:

        if trial > 20:
            break
            
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
#         new_df = new_df.replace([np.inf], None)
#         new_df.dropna(inplace=True)
        
        if new_df.shape[0]<1:
            trial += 1
            continue
#         print('new samples', new_df.shape[0])
        final_df = pd.concat([final_df,new_df]).reset_index(drop=True)
#         print('new samples', new_df.shape[0])
#         print('final samples', final_df.shape[0])
        trial += 1
        
        X2 = new_df.to_numpy()[:,:-1]
        Y2 = new_df.to_numpy()[:,-1]
        dist = euclidean_distances(X2, centroids)
        
        succ_generated += new_df.iloc[np.where((ave_dist>=dist).sum(1)>0)].shape[0]
#         print(trial, succ_generated, trial *  data.shape[0]*generation_coef,  )
    succ_rate = succ_generated/( trial *  data.shape[0]*generation_coef) 
#     print(final_df, succ_rate)
#     input()
#     X2 = new_df.to_numpy()[:,:-1]
#     Y2 = new_df.to_numpy()[:,-1]
#     dist = euclidean_distances(X2, centroids)
    return final_df, succ_rate


for dataset in [ 'Adult']:#, 'Law', 'Student']:

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
    mahND = mahalanobis.MahalanobisND(df.to_numpy().astype(float),10 )
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
            ave_dist.append(mean_dist+ (1 * std_dist))
        elif dataset == 'Student':
            ave_dist.append(mean_dist+ (1 * std_dist))
        else:
            ave_dist.append(mean_dist+(2 * std_dist))
    succ_rate_list =[]
    for Algorithm in ['pc','ges','simy']:
        print('Algorithm', Algorithm)

        for edge_list_filename in glob.glob('./'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
#             print(edge_list_filename)
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
            metrics_temp=[]
            for i in range(1):
                rnd_edges_list = tfb.distributions.Normal(0,1).sample(edges_list.shape[0]).numpy()
                rnd_edges_list = pd.Series(rnd_edges_list, index=edges_list.index).astype('float64')
#                 rnd_edges_list = pd.Series([tfb.distributions.Normal(0,1).sample(1).numpy()[0]]*edges_list.shape[0], index=edges_list.index).astype('float64')
                final_df, succ_rate = generate_dataset_RND(df, graph, rnd_edges_list, ave_dist, KMean.cluster_centers_ ,sens_index, priv_group, unpriv_group)
   
                if final_df.shape[0]>0:
#                     X_rnd = final_df.to_numpy()[:,:-1]
#                     dist = euclidean_distances(X_rnd, KMean.cluster_centers_)
#                     X_org = df.to_numpy()[:,:-1]
#                     X_gen = final_df.to_numpy()[:,:-1]
#                     tree = KDTree(X_org)
#                     distance_list = tree.query(X_gen, k=1)[0]
#                     distance_avg = round(distance_list.mean(),3)
                    
                    distance_avg  = mahND.calc_distances(final_df.to_numpy().astype(float)).mean()
#                     input(dist)
                else:
                    continue

                metrics_temp.append([succ_rate,distance_avg])
    #     succ_rate_list.append([Algorithm,file_num,round(succ_rate,3),distance_avg,distance_std])


            succ_rate_list.append(np.mean(metrics_temp,axis=0))
    input()
    print(f'{dataset} -> ${round(np.mean(succ_rate_list, axis=0)[0],4)}$ & ${round(np.std(succ_rate_list, axis=0)[0],4)}$ & ${round(np.min(succ_rate_list, axis=0)[0],4)}$ & ${round(np.max(succ_rate_list, axis=0)[0],4)}$ & ${round(np.mean(succ_rate_list, axis=0)[1],2)}$')
    
    #     print('Success rate DAG ', Algorithm,'-> Succ_avg = ', round(np.mean(succ_rate_list, axis=0)[0],3),' Dis avg= ', np.mean(succ_rate_list, axis=0)[1])
    np.save('./'+dataset+'_Analysis/RQ1_RND/'+dataset+'_RQ1_results_RND.npy',succ_rate_list)
