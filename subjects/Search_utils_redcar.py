import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.preprocessing import LabelEncoder
import math, time
from fairlearn.metrics import equalized_odds_difference
import tensorflow as tf
import tensorflow_probability as tfb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import xml_parser
import xml_parser_domains
import os,random
from sklearn import tree
from itertools import cycle, islice
tf.random.set_seed(1234)

def DT_tree(df_inp):
    X_DT = df_inp.to_numpy()[:,:-1]
    y_DT = df_inp.to_numpy()[:,-1]
    X_train_DT, X_test_DT, y_train_DT, y_test_DT= train_test_split(X_DT, y_DT,stratify=y_DT)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00']), int(max(y_DT) + 1))))
    color_name = ["blue", "orange"]
    accuracy_max = 0.0
    precision_max = 0.0
    recall_max = 0.0
    rTime_max = 0

    accuracy_avg = 0
    precision_avg = 0
    recall_avg = 0
    clf = None
    alpha = 0.02
    counter = 0
    while True:
        startTime = int(round(time.time() * 1000))
        clf_temp = DecisionTreeClassifier(criterion="gini",splitter='best',max_depth=3, ccp_alpha = alpha)
        clf_temp.fit(X_DT,y_DT)#, sample_weight = sam_weights_train )
        accuracy = clf_temp.score(X_test_DT,y_test_DT)
        y_predict = clf_temp.predict(X_test_DT)
        precision = precision_score(y_test_DT,y_predict,average=None)
        recall = recall_score(y_test_DT,y_predict,average=None)

        endTime = int(round(time.time() * 1000))
        rTime = endTime - startTime
        accuracy_avg = accuracy_avg + accuracy
        precision_avg = precision_avg + precision
        recall_avg = recall_avg + recall
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            precision_max = precision
            recall_max = recall
            clf = clf_temp
            rTime_max = rTime
        depth_tree = clf.get_depth()
        leaves_tree = clf.get_n_leaves()
        if(depth_tree > 1):
            break
        else:
            alpha = 0.0
        counter += 1
        if(counter > 5):
            break
    dot_data = tree.export_graphviz(clf,out_file=None,feature_names=df_inp.columns[:-1], filled=True, rounded = True, impurity = False)
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    edges = graph.get_edge_list()
    for node in nodes:
        if node.get_name() not in ('node', 'edge', '\"\\n\"'):
            values = clf.tree_.value[int(node.get_name())][0]
            if node.get_attributes()['label'].startswith('\"samples'):
                node.set_fillcolor(colors[np.argmax(values)])
            else:

                node.set_fillcolor('whitesmoke')
#             print_out = 'accuracy,precision,recall,total_num_data,total_num_test,tree_computation_time'
#             print(print_out)
#             print(str(accuracy) + "," +  str(precision) + "," + str(recall) + "," + str(len(X_DT)) + "," +  str(len(X_test_DT)) + "," + str(rTime) )   
    
    return graph

def causal_dataset(remove_edge, epsillon ,eliminate , default_acc=0.81  , default_f1=0.50, random_state=0):
    

    df = pd.read_csv('./subjects/datasets/adult_org.csv')
    len_df = df.shape[0]
    df_out=pd.read_csv('./Results/adult_coef1.csv')
    coef_list = ['e0', 'z1e', 'z2e', 'xe', 'ne', 'h0', 'z1h', 'z2h', 'xh', 'nh', 'eh',
           'w0', 'z2w', 'ew', 'nw', 'hw', 'm0', 'z1m', 'z2m', 'wm', 'hm', 'nm',
           'xm', 'o0', 'z1o', 'z2o', 'eo', 'wo', 'mo', 'xo', 'r0', 'mr', 'er',
           'z2r', 'nr', 'xr', 'y0', 'z1y', 'z2y', 'ey', 'oy', 'wy', 'my', 'hy',
           'ry', 'ny', 'xy', 'sigma_h_Sq', 'sigma_h']

    coef_mean = df_out[df_out.columns[8:]].mean()
    edges = {}
    for i in range(len(coef_mean)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df
    #print('Org size', num_samples)
    #num_samples =1000000

    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
    
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsillon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
    else:
        print('No edge selected to remove!')
    for i in range(10):
        
        tf.random.set_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        
        x_prob = df['gender'].sum()/df.shape[0]
        x = tfb.distributions.Bernoulli(x_prob).sample(num_samples).numpy()
        z1_pop = df['race'].unique()
        z1_prob=[]
        for race in z1_pop:
            z1_prob.append(np.where(df['race']==race)[0].shape[0]/df.shape[0])
        z1=np.array(random.choices(z1_pop, weights=z1_prob, k=num_samples))

        z2_pop = df['age'].unique()
        z2_prob=[]
        for age in z2_pop:
            z2_prob.append(np.where(df['age']==age)[0].shape[0]/df.shape[0])
        z2=np.array(random.choices(z2_pop, weights=z2_prob, k=num_samples))

        n_pop = df['native-country'].unique()
        n_prob=[]
        for nc in n_pop:
            n_prob.append(np.where(df['native-country']==nc)[0].shape[0]/df.shape[0])
        n =np.array(random.choices(n_pop, weights=n_prob, k=num_samples))


        e_rate = tf.exp(edges['e0'] + (edges['z1e'] * z1) + (edges['z2e'] * z2) + (edges['ne'] * n) + (edges['xe'] * x))
        
        e =  tfb.distributions.Poisson(rate=e_rate).sample().numpy() 
        h_mean = edges['h0']  + (edges['eh'] * e) + (edges['z1h'] * z1) + (edges['z2h'] * z2) + (edges['nh'] * n) + (edges['xh'] * x)
        h = tfb.distributions.Normal(loc=h_mean, scale= edges['sigma_h']).sample().numpy()

        w_rate = tf.exp(edges['w0']  + (edges['ew'] * e) + (edges['z2w'] * z2) + (edges['nw'] * n) + (edges['hw'] * h))
        w =  tfb.distributions.Poisson(rate=w_rate).sample().numpy()

        m_rate = tf.exp(edges['m0'] + (edges['z1m'] * z1) + (edges['z2m'] * z2) + (edges['nm'] * n) + (edges['xm'] * x) + (edges['hm'] * h) + (edges['wm'] * w))
        m =  tfb.distributions.Poisson(rate=w_rate).sample().numpy()

        o_rate = tf.exp(edges['o0'] + (edges['z1o'] * z1) + (edges['z2o'] * z2) + (edges['eo'] * e) + (edges['xo'] * x) + (edges['mo'] * m) + (edges['wo'] * w))
        o =  tfb.distributions.Poisson(rate=o_rate).sample().numpy()

        r_rate = tf.exp(edges['r0'] + (edges['z2r'] * z2) + (edges['er'] * e) + (edges['xr'] * x) + (edges['mr'] * m) + (edges['nr'] * n))
        r =  tfb.distributions.Poisson(rate=r_rate).sample().numpy()
        y_logits = edges['y0']  + (edges['z1y'] * z1) + (edges['z2y'] * z2) + (edges['ny'] * n) + (edges['ey'] * e) + (edges['hy'] * h) + (edges['wy'] * w) + (edges['my'] * m) + (edges['oy'] * o) + (edges['ry'] * r)
        y = tfb.distributions.Bernoulli(logits = y_logits).sample().numpy()

        new_df = pd.DataFrame(columns = df.columns)
        new_df['age'] = z2
        new_df['workclass'] = w
        new_df['educational-num'] = e
        new_df['marital-status'] = m
        new_df['occupation'] = o
        new_df['relationship'] = r
        new_df['race'] = z1
        new_df['gender'] = x
        new_df['hours-per-week'] = h
        new_df['native-country'] = n
        new_df['label'] = y

        if new_df.isna().sum().sum()!=0:
            print('NA')
            continue
        X_causal = df.to_numpy()[:,:-1]
        Y_causal = df.to_numpy()[:,-1]
        X_train_causal, X_test_causal, y_train_causal, y_test_causal = train_test_split(X_causal, Y_causal,stratify= Y_causal)
        model = LogisticRegression()
        model.fit(X_train_causal,y_train_causal)
        preds_causal = model.predict(X_test_causal)
        acc_temp = accuracy_score(y_test_causal,preds_causal)
        f1_temp = f1_score(y_test_causal,preds_causal)

        if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1 :
#             print('Prediction balance = ',preds.sum()/len(preds), 'test balance = ', y_test.sum()/len(y_test) )

            

            #print(model.score(X_test,y_test), accuracy_score(y_test,preds))
            f1.append(round(f1_temp,2))
            models.append(model)
            dfs.append(new_df.copy())

            test_bal.append(y_test_causal.sum()/len(y_test_causal))
            pred_bal.append(preds_causal.sum()/len(preds_causal))
            break
    
    if len(dfs) > 0:
        return True, dfs[0]
    else:
        return False, None

def causal_pertubation( df_inp,original_program, sensitive_name , causal_edge, step_size, default_acc, eliminate , random_state):
    status, new_df = causal_dataset(remove_edge=causal_edge, epsillon=step_size, eliminate = eliminate, random_state=random_state)
    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        A_new = new_df[sensitive_name].to_numpy()
        X_new = new_df.to_numpy()[:,:-1]
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0)

        df_inp['EOD2'] = 0
        for inp0 in range(df_inp.shape[0]):
            arr_clip = df_inp.iloc[inp0][:15]
            if  arr_clip[3] == 0.0:
                arr_clip[3] =0.0001
            if  arr_clip[6] == 0.0:
                arr_clip[6] = 0.000001

            for feat in range(len(arr_clip)):
                    if 'nan' in str(arr_clip[feat]) or 'none' in str(arr_clip[feat]):
                        arr_clip[feat] = None
                    if type(arr_clip[feat])== float:
                        if math.isnan(arr_clip[feat]):
                            arr_clip[feat] = None
            res1, inp_valid1, score_org, pred_org = original_program(arr_clip, X_train_new, X_test_new, 
                                                                     y_train_new, y_test_new)
            if not res1 :
                print('Traning failed on:')
                print(arr_clip)
                df_inp.loc[inp0,'EOD2'] = "NA"
            elif score_org < default_acc:
                df_inp.loc[inp0,'EOD2'] = "NA"
                #print('Acc low',score_org)

            else: 
                EOD_A = round(equalized_odds_difference(y_test_new, pred_org, sensitive_features=A_test_new ),3)
                df_inp.loc[inp0,'EOD2'] = EOD_A


        return True, df_inp.drop(index=df_inp.loc[df_inp['EOD2']=='NA'].index).reset_index(drop=True)

def nodes_analysis(graph):
    condition = []
    nodes = graph.get_node_list()
    for i in range(len(nodes)):
        if 'label' in nodes[i].obj_dict['attributes']:
            if nodes[int(i)].obj_dict['attributes']['label'].startswith('"samples')==False:
                condition_temp = nodes[int(i)].obj_dict['attributes']['label'].strip().replace('"','').split('\\')[0]
                condition.append(condition_temp)
    return(condition)

def tree_distance(tree1, tree2):
    nodes_tree1 = [x.split(' ')[0] for x in tree1] 
    nodes_tree2 = [x.split(' ')[0] for x in tree2] 
    deletions = [x for x in nodes_tree1 if x not in nodes_tree2]
    additions = [x for x in nodes_tree2 if x not in nodes_tree1]
    return len(deletions) + len(additions)