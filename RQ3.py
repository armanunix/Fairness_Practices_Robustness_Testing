import sys
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

 
    for dataset in ['Compas','Bank','Law','Student','Heart']:
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