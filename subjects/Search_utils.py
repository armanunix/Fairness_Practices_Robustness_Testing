import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.preprocessing import LabelEncoder
import math, time
from fairlearn.metrics import equalized_odds_difference
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os,random
from sklearn import tree
from itertools import cycle, islice




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
#     else:
#         print('No edge selected to remove!')
    for i in range(1):
        
        tf.random.set_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        
        x_prob = df['gender'].sum()/df.shape[0]
        x = tfp.distributions.Bernoulli(x_prob).sample(num_samples).numpy()
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
        
        e =  tfp.distributions.Poisson(rate=e_rate).sample().numpy() 
        h_mean = edges['h0']  + (edges['eh'] * e) + (edges['z1h'] * z1) + (edges['z2h'] * z2) + (edges['nh'] * n) + (edges['xh'] * x)
        h = tfp.distributions.Normal(loc=h_mean, scale= edges['sigma_h']).sample().numpy()

        w_rate = tf.exp(edges['w0']  + (edges['ew'] * e) + (edges['z2w'] * z2) + (edges['nw'] * n) + (edges['hw'] * h))
        w =  tfp.distributions.Poisson(rate=w_rate).sample().numpy()

        m_rate = tf.exp(edges['m0'] + (edges['z1m'] * z1) + (edges['z2m'] * z2) + (edges['nm'] * n) + (edges['xm'] * x) + (edges['hm'] * h) + (edges['wm'] * w))
        m =  tfp.distributions.Poisson(rate=w_rate).sample().numpy()

        o_rate = tf.exp(edges['o0'] + (edges['z1o'] * z1) + (edges['z2o'] * z2) + (edges['eo'] * e) + (edges['xo'] * x) + (edges['mo'] * m) + (edges['wo'] * w))
        o =  tfp.distributions.Poisson(rate=o_rate).sample().numpy()

        r_rate = tf.exp(edges['r0'] + (edges['z2r'] * z2) + (edges['er'] * e) + (edges['xr'] * x) + (edges['mr'] * m) + (edges['nr'] * n))
        r =  tfp.distributions.Poisson(rate=r_rate).sample().numpy()
        y_logits = edges['y0']  + (edges['z1y'] * z1) + (edges['z2y'] * z2) + (edges['ny'] * n) + (edges['ey'] * e) + (edges['hy'] * h) + (edges['wy'] * w) + (edges['my'] * m) + (edges['oy'] * o) + (edges['ry'] * r)
        y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

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
        X_causal = new_df.to_numpy()[:,:-1]
        Y_causal = new_df.to_numpy()[:,-1]
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
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)

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
    
def causal_dataset_redcar_I(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()

    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    r_prob = org_data['race'].sum()/org_data.shape[0]
    r = tfp.distributions.Bernoulli(r_prob).sample(num_samples).numpy()

    a_pop = org_data['age'].unique()
    a_prob=np.zeros(a_pop.shape)
    for age in a_pop:
        a_prob[int(age)] = (np.where(org_data['age']==age)[0].shape[0]/org_data.shape[0])

    a = tfp.distributions.Categorical(probs=a_prob).sample(num_samples).numpy()
    z = tfp.distributions.Normal(0,1).sample(num_samples).numpy()

    #e[j] ~ poisson(exp(e0 + (re * r[j]) + (ae * a[j]) )); 
    e_rate = tf.exp(edges['e0'] + (edges['re'] * r) + (edges['ae'] * a))
    e =  tfp.distributions.Poisson(rate=e_rate).sample().numpy()

    #i[j] ~ normal(i0  + (ai * a[j]) + (ei * e[j]), sigma_h);
    i_mean = edges['i0']  + (edges['ai'] * a) + (edges['ei'] * e) 
    i = tfp.distributions.Normal(loc=i_mean, scale= edges['sigma_h']).sample().numpy()

    #t[j] ~ poisson(exp(t0  + (it * i[j]) + (at * a[j]) + (et * e[j]) + (zt * z[j])));
    t_rate = tf.exp(edges['t0'] + (edges['it'] * i) + (edges['at'] * a) + (edges['et'] * e) + (edges['zt'] * z))
    t =  tfp.distributions.Poisson(rate=t_rate).sample().numpy()

    #x[j] ~ bernoulli_logit(x0 + (zx * z[j]) + (rx * r[j]) + (ax * a[j]) + (tx * t[j]) ); 
    x_logits = edges['x0']  + (edges['zx'] * z) + (edges['rx'] * r) + (edges['ax'] * a) + (edges['tx'] * t) 
    x = tfp.distributions.Bernoulli(logits = x_logits).sample().numpy()

    #y[j] ~ bernoulli_logit(y0  + (zy * z[j]) + (ty * t[j]) + (ay * a[j]));
    y_logits = edges['y0']  + (edges['zy'] * z) + (edges['ty'] * t) + (edges['ay'] * a) 
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

    new_df = pd.DataFrame()
    new_df['aggressive'] = z
    new_df['age'] = a
    new_df['race'] = r
    new_df['education'] = e
    new_df['incom'] = i
    new_df['car_type'] = t
    new_df['redcar'] = x
    new_df['y'] = y
    if new_df.isna().sum().sum()!=0:
        print('NA')
        return False, None, None, None, None, None
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
        return False, None, None, None, None, None
    elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
        return False, None, None, None, None, None    
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
def causal_pertubation_redcar_I(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_redcar_I(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=random_state, drop_feat = None)

    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        A_new = new_df[sensitive].to_numpy()
        X_new = new_df.to_numpy()[:,:-1]
        Y_new = new_df.to_numpy()[:,-1]
        
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
    
    


def causal_dataset_redcar_NI(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()

    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    r_prob = org_data['race'].sum()/org_data.shape[0]
    r = tfp.distributions.Bernoulli(r_prob).sample(num_samples).numpy()

    a_pop = org_data['age'].unique()
    a_prob=np.zeros(a_pop.shape)
    for age in a_pop:
        a_prob[int(age)] = (np.where(org_data['age']==age)[0].shape[0]/org_data.shape[0])

    a = tfp.distributions.Categorical(probs=a_prob).sample(num_samples).numpy()
    z = tfp.distributions.Normal(0,1).sample(num_samples).numpy()

    #e[j] ~ poisson(exp(e0 + (re * r[j]) + (ae * a[j]) )); 
    e_rate = tf.exp((edges['re'] * r) + (edges['ae'] * a))
    e =  tfp.distributions.Poisson(rate=e_rate).sample().numpy()

    #i[j] ~ normal(i0  + (ai * a[j]) + (ei * e[j]), sigma_h);
    i_mean = (edges['ai'] * a) + (edges['ei'] * e) 
    i = tfp.distributions.Normal(loc=i_mean, scale= edges['sigma_h']).sample().numpy()

    #t[j] ~ poisson(exp(t0  + (it * i[j]) + (at * a[j]) + (et * e[j]) + (zt * z[j])));
    t_rate = tf.exp( (edges['it'] * i) + (edges['at'] * a) + (edges['et'] * e) + (edges['zt'] * z))
    t =  tfp.distributions.Poisson(rate=t_rate).sample().numpy()

    #x[j] ~ bernoulli_logit(x0 + (zx * z[j]) + (rx * r[j]) + (ax * a[j]) + (tx * t[j]) ); 
    x_logits = (edges['zx'] * z) + (edges['rx'] * r) + (edges['ax'] * a) + (edges['tx'] * t) 
    x = tfp.distributions.Bernoulli(logits = x_logits).sample().numpy()

    #y[j] ~ bernoulli_logit(y0  + (zy * z[j]) + (ty * t[j]) + (ay * a[j]));
    y_logits = (edges['zy'] * z) + (edges['ty'] * t) + (edges['ay'] * a) 
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

    new_df = pd.DataFrame()
    new_df['aggressive'] = z
    new_df['age'] = a
    new_df['race'] = r
    new_df['education'] = e
    new_df['incom'] = i
    new_df['car_type'] = t
    new_df['redcar'] = x
    new_df['y'] = y
    if new_df.isna().sum().sum()!=0:
        print('NA')
        return False, None, None, None, None, None
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
        return False, None, None, None, None, None
#     elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
#         return False, None, None, None, None, None    
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        if np.unique(y_train_causal_NI).shape[0]==1:
             return False, None, None, None, None, None
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp

def causal_pertubation_redcar_NI(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_redcar_NI(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=random_state, drop_feat = drop_feat)

    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        if drop_feat != None:      
            X_new = new_df.drop(columns=drop_feat).to_numpy()[:,:-1]         
        else:
            X_new = new_df.to_numpy()[:,:-1] 
        A_new = new_df[sensitive].to_numpy()
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
#             elif score_org < default_acc:
#                 df_inp.loc[inp0,'EOD2'] = "NA"
#                 print('Acc low',score_org, default_acc)
#                 input()

            else: 
                EOD_A = round(equalized_odds_difference(y_test_new, pred_org, sensitive_features=A_test_new ),3)
                df_inp.loc[inp0,'EOD2'] = EOD_A
        
        return True, df_inp.drop(index=df_inp.loc[df_inp['EOD2']=='NA'].index).reset_index(drop=True)
    
    
def causal_dataset_adult_I(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    random.seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()
    global new_df
    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    x_prob = org_data['gender'].sum()/org_data.shape[0]
    x = tfp.distributions.Bernoulli(x_prob).sample(num_samples).numpy()

    z1_pop = org_data['race'].unique()
    z1_prob=[]
    for race in z1_pop:
        z1_prob.append(np.where(org_data['race']==race)[0].shape[0]/org_data.shape[0])
    z1=np.array(random.choices(z1_pop, weights=z1_prob, k=num_samples))

    z2_pop = org_data['age'].unique()
    z2_prob=[]
    for age in z2_pop:
        z2_prob.append(np.where(org_data['age']==age)[0].shape[0]/org_data.shape[0])
    z2=np.array(random.choices(z2_pop, weights=z2_prob, k=num_samples))

    n_pop = org_data['native-country'].unique()
    n_prob=[]
    for nc in n_pop:
        n_prob.append(np.where(org_data['native-country']==nc)[0].shape[0]/org_data.shape[0])

    n =np.array(random.choices(n_pop, weights=n_prob, k=num_samples))


    e_rate = tf.exp(edges['e0'] + (edges['z1e'] * z1) + (edges['z2e'] * z2) + (edges['ne'] * n) + (edges['xe'] * x))
    e =  tfp.distributions.Poisson(rate=e_rate).sample().numpy() 

    h_mean = edges['h0']  + (edges['eh'] * e) + (edges['z1h'] * z1) + (edges['z2h'] * z2) + (edges['nh'] * n) + (edges['xh'] * x)
    h = tfp.distributions.Normal(loc=h_mean, scale= edges['sigma_h']).sample().numpy()

    w_rate = tf.exp(edges['w0']  + (edges['ew'] * e) + (edges['z2w'] * z2) + (edges['nw'] * n) + (edges['hw'] * h))
    w =  tfp.distributions.Poisson(rate=w_rate).sample().numpy()

    m_rate = tf.exp(edges['m0'] + (edges['z1m'] * z1) + (edges['z2m'] * z2) + (edges['nm'] * n) + (edges['xm'] * x) + (edges['hm'] * h) + (edges['wm'] * w))
    m =  tfp.distributions.Poisson(rate=w_rate).sample().numpy()

    o_rate = tf.exp(edges['o0'] + (edges['z1o'] * z1) + (edges['z2o'] * z2) + (edges['eo'] * e) + (edges['xo'] * x) + (edges['mo'] * m) + (edges['wo'] * w))
    o =  tfp.distributions.Poisson(rate=o_rate).sample().numpy()

    r_rate = tf.exp(edges['r0'] + (edges['z2r'] * z2) + (edges['er'] * e) + (edges['xr'] * x) + (edges['mr'] * m) + (edges['nr'] * n))
    r =  tfp.distributions.Poisson(rate=r_rate).sample().numpy()

    y_logits = edges['y0']  + (edges['z1y'] * z1) + (edges['z2y'] * z2) + (edges['ny'] * n) + (edges['ey'] * e) + (edges['hy'] * h) + (edges['wy'] * w) + (edges['my'] * m) + (edges['oy'] * o) + (edges['ry'] * r)
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

    new_df = pd.DataFrame(columns = org_data.columns)
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
        return False, None, None, None, None, None
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
        return False, None, None, None, None, None
#     elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
#         return False, None, None, None, None, None    
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        if np.unique(y_train_causal_NI).shape[0]==1:
             return False, None, None, None, None, None
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp

def causal_pertubation_adult_I(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_adult_I(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=0, drop_feat = drop_feat)


    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        if drop_feat != None:      
            X_new = new_df.drop(columns=drop_feat).to_numpy()[:,:-1]         
        else:
            X_new = new_df.to_numpy()[:,:-1] 
        A_new = new_df[sensitive].to_numpy()
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
#             elif score_org < default_acc:
#                 df_inp.loc[inp0,'EOD2'] = "NA"
#                 print('Acc low',score_org, default_acc)
#                 input()

            else: 
                EOD_A = round(equalized_odds_difference(y_test_new, pred_org, sensitive_features=A_test_new ),3)
                df_inp.loc[inp0,'EOD2'] = EOD_A
        
        return True, df_inp.drop(index=df_inp.loc[df_inp['EOD2']=='NA'].index).reset_index(drop=True)
    

def causal_dataset_compas_NI(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    random.seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()

    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    r_prob = org_data['race'].sum()/org_data.shape[0]
    r = tfp.distributions.Bernoulli(r_prob).sample(num_samples).numpy()

    s_prob = org_data['sex'].sum()/org_data.shape[0]
    s = tfp.distributions.Bernoulli(s_prob).sample(num_samples).numpy()

    a_pop = org_data['age_cat'].unique()
    a_prob=[]
    for age in a_pop:
        a_prob.append(np.where(org_data['age_cat']==age)[0].shape[0]/org_data.shape[0])
    a=np.array(random.choices(a_pop, weights=a_prob, k=num_samples))

#     j[i] ~ poisson(exp((rj * r[i]) + (sj * s[i]) + (aj * a[i])) ); 
    j_rate = tf.exp((edges['rj'] * r) + (edges['sj'] * s) + (edges['aj'] * a) )
    j =  tfp.distributions.Poisson(rate=j_rate).sample().numpy() 
    
#     p[i] ~ poisson(exp((rp * r[i]) + (sp * s[i]) + (ap * a[i]) + (jp * j[i])));
    p_rate = tf.exp((edges['rp'] * r) + (edges['sp'] * s) + (edges['ap'] * a) + (edges['jp'] * j))
    p =  tfp.distributions.Poisson(rate=p_rate).sample().numpy() 
    
#    d[i] ~ poisson(exp( (rd * r[i]) + (sd * s[i]) + (ad * a[i]) + (pd * p[i]) + (jd * j[i]) )); 
    d_rate = tf.exp((edges['rd'] * r) + (edges['sd'] * s) + (edges['ad'] * a) + (edges['jd'] * j) + (edges['pd'] * p) )
    d =  tfp.distributions.Poisson(rate=d_rate).sample().numpy() 
    
#    y[i] ~ bernoulli_logit( (ry * r[i]) + (sy * s[i]) + (ay * a[i]) + (dy * d[i]) + (jy * j[i]) );


    y_logits = (edges['ry'] * r) + (edges['sy'] * s) + (edges['ay'] * a) + (edges['dy'] * d) + (edges['jy'] * j)
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()
    

    new_df = pd.DataFrame(columns = org_data.columns)
    new_df['sex'] = s
    new_df['race'] = r
    new_df['age_cat'] = a
    new_df['priors_count'] = p
    new_df['juv_fel_count'] = j
    new_df['r_charge_degree'] = d
    new_df['label'] = y
    
    if new_df.isna().sum().sum()!=0:
        print('NA')
        return False, None, None, None, None, None
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
    elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
        return False, None, None, None, None, None    
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp

def causal_pertubation_compas_NI(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_compas_NI(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=random_state, drop_feat = drop_feat)


    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        if drop_feat != None:      
            X_new = new_df.drop(columns=drop_feat).to_numpy()[:,:-1]         
        else:
            X_new = new_df.to_numpy()[:,:-1] 
        A_new = new_df[sensitive].to_numpy()
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
#             elif score_org < default_acc:
#                 df_inp.loc[inp0,'EOD2'] = "NA"
#                 print('Acc low',score_org, default_acc)
#                 input()

            else: 
                EOD_A = round(equalized_odds_difference(y_test_new, pred_org, sensitive_features=A_test_new ),3)
                df_inp.loc[inp0,'EOD2'] = EOD_A
        
        return True, df_inp.drop(index=df_inp.loc[df_inp['EOD2']=='NA'].index).reset_index(drop=True)
    

def causal_dataset_law(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    random.seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()
    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    
    
    s_prob = org_data['sex'].sum()/org_data.shape[0]
    s = tfp.distributions.Bernoulli(s_prob).sample(num_samples).numpy()
    r_pop = org_data['race'].unique()
    r_prob=[]
    for race in r_pop:
        r_prob.append(np.where(org_data['race']==race)[0].shape[0]/org_data.shape[0])
        #r_prob.append(0.125)

    r=np.array(random.choices(r_pop, weights=r_prob, k=num_samples))

    
    
#     g[i] ~ normal(g0 + (rg * r[i]) + (sg * s[i]) , sigma_g);
    g_mean = edges['g0']  + (edges['rg'] * r) + (edges['sg'] * s)
    g = tfp.distributions.Normal(loc=g_mean, scale= edges['sigma_g']).sample().numpy()
    
#     l[i] ~ poisson(exp(l0 + (rl * r[i]) + (sl * s[i]) ));
    
    l_rate = tf.exp(edges['l0'] + (edges['rl'] * r) + (edges['sl'] * s))
    l =  tfp.distributions.Poisson(rate=l_rate).sample().numpy() 
    
#     y[i] ~ bernoulli_logit((ry * r[i]) + (sy * s[i]) );
    

    y_logits =(edges['ry'] * r) + (edges['sy'] * s) 
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

#     if y.sum()/len(y)<0.2 or y.sum()/len(y)>0.89:
#         print('Bal',  y.sum()/len(y))
#         continue
    new_df = pd.DataFrame(columns = org_data.columns)
    new_df['race'] = r
    new_df['sex'] = s
    new_df['LSAT'] = l
    new_df['UGPA'] = g
    new_df['label'] = y

    if new_df.isna().sum().sum()!=0:
        print('NA')
        return False, None, None, None, None, None
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
#     elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
#         return False, None, None, None, None, None    
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp

def causal_pertubation_law(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_law(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=0, drop_feat = drop_feat)


    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        if drop_feat != None:      
            X_new = new_df.drop(columns=drop_feat).to_numpy()[:,:-1]         
        else:
            X_new = new_df.to_numpy()[:,:-1] 
        A_new = new_df[sensitive].to_numpy()
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
#             elif score_org < default_acc:
#                 df_inp.loc[inp0,'EOD2'] = "NA"
#                 print('Acc low',score_org, default_acc)
#                 input()

            else: 
                EOD_A = round(equalized_odds_difference(y_test_new, pred_org, sensitive_features=A_test_new ),3)
                df_inp.loc[inp0,'EOD2'] = EOD_A
        
        return True, df_inp.drop(index=df_inp.loc[df_inp['EOD2']=='NA'].index).reset_index(drop=True)
        
def causal_dataset_student(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, 
                              default_acc=None  , default_f1=None, random_state=0, drop_feat = None):
    tf.random.set_seed(random_state)
    len_df = org_data.shape[0]
    coef_mean = coef[coef.columns[8:]].mean()
    global new_df
    edges = {}
    for i in range(len(coef_list)):
        edges[coef_list[i]] = coef_mean[i]

    num_samples =len_df


    models =[]
    dfs =[]
    acc =[]
    f1  =[]
    test_bal =[]
    pred_bal =[]
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
    if remove_edge!=None and eliminate == False:
        edges[remove_edge] =  edges[remove_edge] + epsilon 
    elif remove_edge!=None and eliminate == True:
        edges[remove_edge] =  0
#     else:
#         print('No edge selected to remove!')
#     if remove_edge!=None:
#         print('before',edges[remove_edge])
#     if remove_edge=='re':   
#         print('re',edges[remove_edge]) 
    
    
    s_prob = org_data['sex'].sum()/org_data.shape[0]
    s = tfp.distributions.Bernoulli(s_prob).sample(num_samples).numpy()
    
#    g1[i] ~ poisson(exp((sg1 * s[i]) + g10 )); 
    g1_rate = tf.exp(edges['g10'] + (edges['sg1'] * s) )
    g1 =  tfp.distributions.Poisson(rate=g1_rate).sample().numpy() 

#     g2[i] ~ poisson(exp((sg2 * s[i])+ (g1g2 * g1[i]) + g20));
    g2_rate = tf.exp(edges['g20'] + (edges['sg2'] * s)  + (edges['g1g2'] * g1))
    g2 =  tfp.distributions.Poisson(rate=g2_rate).sample().numpy()
    
#     f[i] ~ poisson(exp((sf * s[i]) + f0 ));
    f_rate = tf.exp(edges['f0'] + (edges['sf'] * s) )
    f =  tfp.distributions.Poisson(rate=f_rate).sample().numpy()
    
#     h[i] ~ bernoulli_logit((g1h * g1[i])+ (fh * f[i])+ (g2h * g2[i]) + h0  );
    h_logits = edges['h0']  + (edges['g1h'] * g1) + (edges['fh'] * f) + (edges['g2h'] * g2)
    h = tfp.distributions.Bernoulli(logits = h_logits).sample().numpy()
        
#     y[i] ~ bernoulli_logit( (g1y * g1[i])+ (fy * f[i])+ (g2y * g2[i]) + y0  );
    
    y_logits = edges['y0']  + (edges['g1y'] * g1) + (edges['fy'] * f) + (edges['g2y'] * g2)
    y = tfp.distributions.Bernoulli(logits = y_logits).sample().numpy()

    

   

    new_df = pd.DataFrame(columns = org_data.columns)
    new_df['sex'] = s
    new_df['failures'] = f
    new_df['higher'] = h
    new_df['G1'] = g1
    new_df['G2'] = g2
    new_df['label'] = y

    if new_df.isna().sum().sum()!=0:
        print('NA')
        return False, None, None, None, None, None
#     elif y.sum()/y.shape[0] <0.1 or y.sum()/y.shape[0] >0.9:
#         return False, None, None, None, None, None  
    elif np.isinf(new_df).sum().sum()>0:
        print('inf')
        return False, None, None, None, None, None
    else:
        if drop_feat != None:      
            X_causal_NI = new_df.drop(columns=drop_feat).to_numpy()[:,:-1] 

        else:
            X_causal_NI = new_df.to_numpy()[:,:-1] 

        Y_causal_NI = new_df.to_numpy()[:,-1]
        A_causal_NI = new_df[sensitive].to_numpy()
        
        if np.where(Y_causal_NI==0)[0].shape[0]<3 or np.where(Y_causal_NI==1)[0].shape[0]<3 :
            return False, None, None, None, None, None
        X_train_causal_NI, X_test_causal_NI, y_train_causal_NI, y_test_causal_NI , A_train_causal_NI, A_test_causal_NI = train_test_split(X_causal_NI, Y_causal_NI,A_causal_NI,stratify= Y_causal_NI, random_state=0)
        model = LogisticRegression()
        if np.unique(y_train_causal_NI).shape[0]==1:
            return False, None, None, None, None, None
        model.fit(X_train_causal_NI,y_train_causal_NI)
        model.fit(X_train_causal_NI,y_train_causal_NI)
        preds_causal_NI = model.predict(X_test_causal_NI)
        acc_temp = accuracy_score(y_test_causal_NI,preds_causal_NI)
        f1_temp = f1_score(y_test_causal_NI,preds_causal_NI)
        EOD = round(equalized_odds_difference(y_test_causal_NI, preds_causal_NI, sensitive_features=A_test_causal_NI ),3)
#         print('causal dataset',round(acc_temp,2),round(f1_temp,2))
#         print('causal dataset',default_acc,default_f1)
        if default_acc==None or default_f1==None:

            return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            
        else:
            
            if round(acc_temp,2) ==default_acc and  round(f1_temp,2)==default_f1:
                

                return True, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp
            else:

                return False, new_df, preds_causal_NI.sum()/len(preds_causal_NI), EOD, acc_temp, f1_temp

def causal_pertubation_student(org_data,original_program, df_inp, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc  , default_f1, random_state=0, drop_feat = None):
    
    status, new_df, preds_bal, EOD, acc, f1 = causal_dataset_student(org_data, coef, coef_list, remove_edge, epsilon ,eliminate,  sensitive, default_acc=default_acc  , default_f1=default_f1, random_state=random_state, drop_feat = drop_feat)


    if status == False:
        return False, None
    else:
         #new_df.to_csv('Results/'+program_name+'/'+dataset+'/'+sensitive_name+'/input_test//'+str(causal_edge)+'_data.csv',index=False)
        if drop_feat != None:      
            X_new = new_df.drop(columns=drop_feat).to_numpy()[:,:-1]         
        else:
            X_new = new_df.to_numpy()[:,:-1] 
        A_new = new_df[sensitive].to_numpy()
        Y_new = new_df.to_numpy()[:,-1]
        X_train_new, X_test_new, y_train_new, y_test_new, A_train_new, A_test_new = train_test_split(X_new, Y_new, A_new, random_state=0,stratify=Y_new)
        
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
#             elif score_org < default_acc:
#                 df_inp.loc[inp0,'EOD2'] = "NA"
#                 print('Acc low',score_org, default_acc)
#                 input()

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