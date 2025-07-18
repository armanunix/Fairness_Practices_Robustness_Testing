import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Select the dataset')
parser.add_argument('--practice', type=str, help='Select the practice to be tested')
args = parser.parse_args()
dataset=args.dataset
Practice=args.practice


if Practice in ['SelectKBest', 'SelectFpr','SelectPercentile' ,'drop']:
    import warnings
    warnings.filterwarnings('ignore')
    import sys,os
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
    import glob
    import re
    
        
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
        df = pd.read_csv('./subjects/datasets/'+data_file_name)
        df = df.drop_duplicates()
    
        X1 = df.to_numpy()[:,:-1]
        Y1 = df.to_numpy()[:,-1].astype(int)
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
            ave_dist.append(mean_dist + (2 * std_dist))
            #ave_dist.append(euclidean_distances(X1[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).max())
    
        for Algorithm in ['pc','ges','simy']:
            print('Algorithm', Algorithm)
            for edge_list_filename in glob.glob('./'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
                print(edge_list_filename)
                file_num = int(re.findall(r'\d+', edge_list_filename)[0])
                RQ1_res = np.load('./'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy')
                if RQ1_res[np.where(RQ1_res[:,1].astype(int)==file_num)[0]][0][2].astype(float)==0.0:
                    continue
                try:
                    graph_filename = './'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
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
                        for transformer in [Practice]:
    
    
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
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_EOD_' + str(file_num)+'.npy',SelectKBest_final_EOD)
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectKbest_AOD_' + str(file_num)+'.npy',SelectKBest_final_AOD)
    
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_EOD_' + str(file_num)+'.npy',SelectFpr_final_EOD)
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectFpr_AOD_' + str(file_num)+'.npy',SelectFpr_final_AOD)
    
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_EOD_' + str(file_num)+'.npy',SelectPercentile_final_EOD)
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_SelectPercentile_AOD_' + str(file_num)+'.npy',SelectPercentile_final_AOD)
    
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_EOD_' + str(file_num)+'.npy',drop_final_EOD)
                np.save('./'+dataset+'_Analysis/RQ2/'+Algorithm+'_drop_AOD_' + str(file_num)+'.npy',drop_final_AOD)

if Practice in ['TO','CEO']:
    import warnings
    warnings.filterwarnings('ignore')
    import sys,os
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
            
        df = pd.read_csv('./subjects/datasets/'+data_file_name)
        df =  df.drop_duplicates().reset_index(drop=True)
    
        X1 = df.to_numpy()[:,:-1]
        Y1 = df.to_numpy()[:,-1].astype(int)
    
        
        for Algorithm in alg_list:
            print('Algorithm', Algorithm)
            for edge_list_filename in glob.glob('./'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
                print(edge_list_filename)
                file_num = int(re.findall(r'\d+', edge_list_filename.split('/')[-1])[0])
                RQ1_res = np.load('./'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy')
                if RQ1_res[np.where(RQ1_res[:,1].astype(int)==file_num)[0]][0][2].astype(float)==0.0:
                    print('No',file_num)
                    continue
                try:
                    graph_filename = './'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
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
                    
                np.save('./'+dataset+'_Analysis/RQ4/'+Algorithm+'_'+Practice +'_'+ str(file_num)+'.npy',mitigator_final_EOD)
                np.save('./'+dataset+'_Analysis/RQ4/'+Algorithm+'_'+Practice +'_' + str(file_num)+'.npy',mitigator_final_AOD)






if Practice in ['HP']:
    import sys,os
    sys.path.append("./subjects/")
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    import pandas as pd
    import os
    import time
    import re, random
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNetCV
    from sklearn.cluster import KMeans
    import shap
    import xml_parser
    import xml_parser_domains
    from mutation import mutate, clip_LR
    from fairlearn.metrics import equalized_odds_difference
    from sklearn.preprocessing import LabelEncoder
    from Utils_Functions import generate_dataset, eod
    import copy, pickle, glob
    from sklearn.metrics.pairwise import  euclidean_distances
    import argparse
    
    
    
    def HP_search_population(X, Y, A,X_shap, HP_population, program_name, max_iter, 
                   sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, start_time):
        time1 = time.time()
        df_input = pd.DataFrame()
        
        if(program_name == "LogisticRegression"):
            import Logistic_Regression_Mitigation
            original_program = Logistic_Regression_Mitigation.logisticRegressionOriginal
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
        
    
        for arr_clip in HP_population:
    
            res1, LR, inp_valid1, score_org, pred_org = original_program(arr_clip, X, X, Y, Y)
    
    
            EOD_A = eod(Y, pred_org,sens=A, priv=priviliged_group, unpriv=unpriviliged_group)
            
            
            df_input = df_input.append([EOD_A])
            #print("---------------------------------------------------------")                 
            
            #print('Iteration took ', time.time() - time2)
    
        
    
        df_input.columns=['EOD']
        #label = np.digitize(df_input['EOD'],bins=bins)
        Y_shap = df_input['EOD']
        ENCV =  ElasticNetCV(cv=10).fit(X_shap,Y_shap) 
        explainer = shap.LinearExplainer(ENCV, X_shap)
        shap_values = explainer.shap_values(X_shap)
        HP_importance = X_shap.columns[np.argsort(np.abs(shap_values).mean(0))[::-1]]
        return HP_importance, df_input, shap_values
    
    
    def HP_search(X, Y, A, dataset, program_name, max_iter, 
                   sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, start_time):
        time1 = time.time()
        df_input = pd.DataFrame()
        default_acc = 0.0
        HP_population = []
        
          
        if(program_name == "LogisticRegression"):
            import Logistic_Regression_Mitigation
            original_program = Logistic_Regression_Mitigation.logisticRegressionOriginal
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
        
    
        arr_min, arr_max, arr_type, arr_default = xml_parser_domains.xml_parser_domains(input_program_tree, num_args)
    
        seeds = []
        promising_inputs_EOD = []
        promising_metric_EOD = []
    
       
        EOD_max = 0
        for counter in range(max_iter):
            time2 = time.time()
            #print('counter',counter)  
    
            inp = mutate( arr_max, arr_min, arr_type, arr_default, promising_inputs_EOD, counter)
    
            if re.match(r'Logistic', program_name):
                arr_clip, features = clip_LR(inp)
            
            if arr_clip in seeds:
                #print('Duplicated Seed!')
                continue
            else:
                seeds.append(arr_clip)
            EOD_avg = []
    
            
            X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, Y, A)
            res1, LR, inp_valid1, score_org, pred_org = original_program(arr_clip, X_train, X_test, 
                                                                                y_train, y_test)
            if not res1:
                #print('training failed')
                continue
            else:
                failed_flag = False
    
            # tolerate at most 2% of accuracy loss        
            if counter == 0:
                default_acc = score_org
    
            if (score_org < (default_acc - 0.05)) :
                #print('Acc low')
                continue
            res1, LR1, inp_valid1, score_org, pred_org = original_program(arr_clip, X, X, Y, Y)
            EOD_A = eod(Y, pred_org,sens=A, priv=priviliged_group, unpriv=unpriviliged_group)
    #         pred = LR.predict(X)
    #         # A original model , B mitigated mode   
    #         EOD_avg.append(eod(Y, pred,sens=A, priv=priviliged_group, unpriv=unpriviliged_group))
    #         print(EOD_avg[-1])
    #         input()
    #         EOD_A = np.mean(EOD_avg)
    #         print(EOD_A)
    #         print('------')
            #print(eod_test_A, EOD_A)
            if abs(EOD_A) > EOD_max:
                EOD_max = abs(EOD_A)
                #print('Intresting EOD')
                intresting_EOD = 1
                promising_inputs_EOD.append(inp)
            elif abs(EOD_A) > 0.1:
                #print('Intresting EOD')
                intresting_EOD = 1
                promising_inputs_EOD.append(inp)
            else:
                intresting_EOD = 0        
            df_input = pd.concat([df_input,pd.DataFrame(arr_clip + [EOD_A]).T] , sort=False)
            HP_population.append(arr_clip)
            #print("---------------------------------------------------------")                 
            if time.time() - time1 >= time_out:
                break
            #print('Iteration took ', time.time() - time2)
    
        
        df_input.columns = ['solver', 'penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 
                            'max_iteration', 'multi_class', 'l1_ratio', 'class_weight', 'random_state', 'verbose',
                            'warm_start', 'n_jobs','EOD']
        df_input.reset_index(drop=True,inplace=True)
        df_input.to_csv('./'+dataset+'_Analysis/RQ3/'+ dataset + '_base_HP_search.csv', index=False)
        ind_0 = df_input.sort_values(by='EOD').head(200).index.to_list()
        ind_1 = df_input.sort_values(by='EOD').tail(200).index.to_list()
        HP_population = np.array(HP_population)[ind_0 + ind_1]
        np.save('./'+dataset+'_Analysis/RQ3/'+ dataset + '_HP_population.csv',HP_population)
        df_inp = df_input.iloc[ind_0+ind_1]
    #     df_inp.loc[ind_0,'EOD'] = 0
    #     df_inp.loc[ind_1,'EOD'] = 1
        df_inp.reset_index(drop=True,inplace=True)
        le =  LabelEncoder()
        cat_columns = ['solver', 'penalty', 'dual','fit_intercept', 'multi_class', 'warm_start']
        for col in cat_columns:
            df_inp[col] = le.fit_transform(df_inp[col])
        columns = []
        for col in df_inp.columns[:-1]:
            if df_inp[col].unique().shape[0] > 1 :
                columns.append(col)           
        label = df_inp['EOD'].to_numpy()
    #     bins = np.histogram(df_inp['EOD'], bins=3)[1][1:-1].tolist()
    #     label = np.digitize(df_inp['EOD'],bins=bins)
    
        df_inp = df_inp[columns]
        for ind in np.where(df_inp.isna().sum()>0):
            df_inp[df_inp.columns[ind]+'_isna']=0
            df_inp.loc[np.where(df_inp[df_inp.columns[ind]].isna()==True)[0],df_inp.columns[ind]+'_isna'] = 1
            df_inp.loc[np.where(df_inp[df_inp.columns[ind]].isna()==True)[0], df_inp.columns[ind]] = 0
        df_inp['label'] = label
        X_shap = df_inp[df_inp.columns[:-1]]
        Y_shap = df_inp['label']
        features = X_shap.columns
        cat_features = []
        for cat in X_shap.select_dtypes(exclude="number"):
            cat_features.append(cat)
            X_shap[cat] = X_shap[cat].astype("category").cat.codes.astype("category")
        ENCV =  ElasticNetCV(cv=10).fit(X_shap,Y_shap) 
        explainer = shap.LinearExplainer(ENCV, X_shap)
        shap_values = explainer.shap_values(X_shap)
        HP_importance = X_shap.columns[np.argsort(np.abs(shap_values).mean(0))[::-1]]
        return HP_importance, df_input, HP_population, X_shap
    if __name__ == '__main__':
        start_time = time.time()
        time_out = 60 * 60
        original_model = True
        save_model = False
        num_iteration =  1000000
        program_name="LogisticRegression"
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, help='Select the dataset')
        args = parser.parse_args()
        dataset=args.dataset
        
        for dataset in [dataset]:
            HP_importance_results = []
            final_shap_list = []
            #data_config.keys(): 
            if dataset == 'Adult':
                sensitive_param = 7
                sensitive_name = 'gender'
                priviliged_group = 1  #male
                unpriviliged_group = 0#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'adult_org-Copy1.csv'
                graph_base_filename ='./Adult_Analysis/ges/DAGs/Adult_ges_DAG_1.csv'
                edge_list_base_filename = './Adult_Analysis/ges/PP/Adult_ges_pp_1.csv'
                alg_list = ['ges','simy']
            if dataset == 'Compas':
                sensitive_param = 1
                sensitive_name = 'race'
                priviliged_group = 1  #male
                unpriviliged_group = 0#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'compas-Copy1'
                graph_base_filename ='./Compas_Analysis/pc/DAGs/Compas_pc_DAG_13.csv'
                edge_list_base_filename = './Compas_Analysis/pc/PP/Compas_pc_pp_13.csv'
                alg_list = ['ges','pc']
            if dataset == 'Bank':
                sensitive_param = 0
                sensitive_name = 'age'
                priviliged_group = 5  #male
                unpriviliged_group = 3#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'bank'
                graph_base_filename ='./Bank_Analysis/ges/DAGs/Bank_ges_DAG_8.csv'
                edge_list_base_filename = './Bank_Analysis/ges/PP/Bank_ges_pp_8.csv'
                alg_list = ['ges']
            if dataset == 'Law':
                sensitive_param = 0
                sensitive_name = 'sex'
                priviliged_group = 1  #male
                unpriviliged_group = 0#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'law.csv'
                graph_base_filename ='./Law_Analysis/ges/DAGs/Law_ges_DAG_15.csv'
                edge_list_base_filename = './Law_Analysis/ges/PP/Law_ges_pp_15.csv'
                alg_list = ['ges','simy']
            if dataset == 'Student':
                sensitive_param = 0
                sensitive_name = 'sex'
                priviliged_group = 1  #male
                unpriviliged_group = 0#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'students-processed_2'
                graph_base_filename ='./Student_Analysis/pc/DAGs/Student_pc_DAG_1.csv'
                edge_list_base_filename = './Student_Analysis/pc/PP/Student_pc_pp_1.csv'
                alg_list = ['simy','pc']
            if dataset == 'Heart':
                sensitive_param = 0
                sensitive_name = 'sex'
                priviliged_group = 1  #male
                unpriviliged_group = 0#female
                favorable_label  = 1.0
                unfavorable_label = 0.0
                data_file_name = 'heart_processed_1'
                graph_base_filename ='./Heart_Analysis/ges/DAGs/Heart_ges_DAG_4.csv'
                edge_list_base_filename = './Heart_Analysis/ges/PP/Heart_ges_pp_4.csv'
                alg_list = ['ges']
            df = pd.read_csv('./subjects/datasets/'+data_file_name)
    
            df = df.drop_duplicates().reset_index(drop=True)
            A = df[sensitive_name].to_numpy()
            X = df.to_numpy()[:,:-1]
            Y = df.to_numpy()[:,-1].astype(int)
    
            num_cluster = 100
            try :
                with open('./'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster), 'rb') as f:
                    KMean = pickle.load(f)
            except:
                KMean = KMeans(n_clusters=num_cluster)
                KMean.fit(X)
                with open('./'+dataset+'_Analysis/Kmean/KMean_{clus}.pkl'.format(clus=num_cluster),'wb') as f:
                    pickle.dump(KMean,f)
    
            ave_dist =[] 
            for i in range(KMean.n_clusters):
                mean_dist = euclidean_distances(X[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).mean()
                std_dist = euclidean_distances(X[np.where(KMean.labels_==[i])],[KMean.cluster_centers_[i]]).std()
                ave_dist.append(mean_dist + (2 * std_dist))
    
    
    
    
            graph_base = pd.read_csv(graph_base_filename)
            edges_list_base = pd.read_csv(edge_list_base_filename)
    
            edges_base = edges_list_base[edges_list_base.columns[1:-1]].mean()
            final_df_base, succ_rate_base = generate_dataset(df, graph_base, edges_base, ave_dist, KMean.cluster_centers_ ,sensitive_param, priviliged_group, unpriviliged_group)
                                
            A_base = final_df_base[sensitive_name].to_numpy()
            X_base = final_df_base.to_numpy()[:,:-1]
            Y_base = final_df_base.to_numpy()[:,-1].astype(int)
            HP_importance_org , df_HP_org, HP_population, X_shap_org = HP_search(X_base, Y_base, A_base, dataset, program_name, num_iteration, 
                       sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, 
                       start_time)
    
            HP_importance_results.append([dataset,'original', 'original', 'original'] + HP_importance_org.to_list())
            
            final_shap_list.append(X_shap_org)
            for Algorithm in alg_list:
                print('Algorithm', Algorithm)
                for edge_list_filename in glob.glob('./'+dataset+'_Analysis/'+Algorithm+'/PP/*.csv'):
                    print(edge_list_filename)
                    file_num = int(re.findall(r'\d+', edge_list_filename)[0])
                    RQ1_res = np.load('./'+dataset+'_Analysis/RQ1/'+dataset+'_'+Algorithm+'_RQ1_results.npy')
                    if RQ1_res[np.where(RQ1_res[:,1].astype(int)==file_num)[0]][0][2].astype(float)==0.0:
                        print('No',file_num)
                        continue
                    try:
                        graph_filename = './'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num)
                        graph = pd.read_csv(graph_filename)
    
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
    
                    for weights_num in range(KMean_coef.n_clusters):
    
                        #print('Coef ',i)
                        weights_ind = np.random.choice(np.where(KMean_coef.labels_==weights_num)[0])
                        edges = edges_list.iloc[weights_ind]
                        
                       
                        final_df, succ_rate = generate_dataset(df, graph, edges, ave_dist, KMean.cluster_centers_ ,sensitive_param, priviliged_group, unpriviliged_group)
                        if succ_rate!=0.0:
                            A1 = final_df[sensitive_name].to_numpy()
                            X1 = final_df.to_numpy()[:,:-1]
                            Y1 = final_df.to_numpy()[:,-1].astype(int)
    
    
                            HP_importance_pert, df_HP_pert ,shap_values= HP_search_population(X1, Y1, A1, X_shap_org, HP_population, program_name, num_iteration, 
                                       sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, 
                                       start_time)
    
                            final_shap_list.append(shap_values)
                            HP_importance_results.append([dataset,Algorithm, file_num, weights_ind] + HP_importance_pert.to_list())
                        else:
                            print('Zero succ')
    
    
            np.save('./'+dataset+'_Analysis/RQ3/'+ dataset + '_RQ3.npy', HP_importance_results)
            np.save('./'+dataset+'_Analysis/RQ3/'+ dataset + '_RQ3_shap_values.npy', final_shap_list)