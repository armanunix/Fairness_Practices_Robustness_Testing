import pandas as pd
import numpy as np
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Select the dataset(Adult, Bank, ...)')
parser.add_argument('--alg', type=str, help='Select the discovery algorithm(pc, ges, simmy)')
args = parser.parse_args()
dataset=args.dataset
Algorithm=args.alg
if dataset=='Bank':
    data = pd.read_csv('../subjects/datasets/Bank')
elif dataset=='Adult':
    data = pd.read_csv('../subjects/datasets/adult_org-Copy1.csv')
elif dataset=='Compas':
    data = pd.read_csv('../subjects/datasets/compas-Copy1')
elif dataset=='Heart':
    data = pd.read_csv('../subjects/datasets/heart_processed_1')
elif dataset=='Law':
    data = pd.read_csv('../subjects/datasets/law.csv')
elif dataset=='Student':
    data = pd.read_csv('../subjects/datasets/students-processed_2')

dataset_types = [str(data[i].dtype) for i in data.columns]
for file_num in range(1,100):
    #file_num =107
    not_a_DAG = False
    graph = pd.read_csv('../'+dataset+'_Analysis/'+Algorithm+'/DAGs/'+dataset+'_'+Algorithm+'_DAG_{file_num}.csv'.format(file_num=file_num))
    graph.columns =  [graph.columns[0]]+ data.columns.to_list()
    graph[graph.columns[0]] = data.columns.to_list()
    statring_atts = graph.sum().index[np.where(graph.sum()==0)[0]]
    data_to_file = []
#    with open('./pc/PP/Adult_pc_DAG_{file_num}.stan'.format(file_num=file_num),'w') as file:
    #file.write('data{\n')
    data_to_file.append('data{\n')

    #file.write('int<lower = 0> N;\n')
    data_to_file.append('int<lower = 0> N;\n')
    for att in graph.columns[1:]:
        #file.write('array[N] int<lower={min_bound}, upper={max_bound}>  {x};\n'.format(min_bound = data.min()[att], max_bound = data.max()[att], x = att))
        if 'float' in dataset_types[np.where(data.columns==att)[0][0]]:
            data_to_file.append('array[N] real<lower={min_bound}, upper={max_bound}>  {x};\n'.format(min_bound = data.min()[att], max_bound = data.max()[att], x = att))

        else:           
            data_to_file.append('array[N] int<lower={min_bound}, upper={max_bound}>  {x};\n'.format(min_bound = data.min()[att], max_bound = data.max()[att], x = att))
    #file.write('}\n')
    
        
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    #file.write('transformed data {\n')
    data_to_file.append('transformed data {\n')
    #file.write('}\n')
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    #file.write('parameters {\n')
    data_to_file.append('parameters {\n')
    need_posterier=[]
    for ind in range(graph.shape[0]):
        for att in graph.columns[np.where(graph.iloc[ind]==1)[0]]:
            #file.write('real {x}{y};\n'.format(x = graph['Unnamed: 0'][ind], y = att))
            data_to_file.append('real {x}{y};\n'.format(x = graph['Unnamed: 0'][ind], y = att))

        if graph['Unnamed: 0'][ind] not in statring_atts:
            #file.write('real {x}0;\n'.format(x = graph['Unnamed: 0'][ind]))
            data_to_file.append('real {x}0;\n'.format(x = graph['Unnamed: 0'][ind]))
            need_posterier.append(graph['Unnamed: 0'][ind])

        #file.write('\n')
        data_to_file.append('\n')
    if 0 in np.char.find(dataset_types,'float'):
        data_to_file.append('real<lower=0> sigma_h_Sq;\n')
    #file.write('}\n')
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    #file.write('transformed parameters {\n')
    data_to_file.append('transformed parameters {\n')
    #file.write('}\n')
    if 0 in np.char.find(dataset_types,'float'):
        data_to_file.append('real<lower=0> sigma_h;\n')
        data_to_file.append('sigma_h = sqrt(sigma_h_Sq);\n')
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    #file.write('model {\n')
    data_to_file.append('model {\n')

    for ind in range(graph.shape[0]):
        for att in graph.columns[np.where(graph.iloc[ind]==1)[0]]:
            #file.write('{x}{y}        ~ normal(0, 1);\n'.format(x = graph['Unnamed: 0'][ind], y = att))
            data_to_file.append('{x}{y}        ~ normal(0, 1);\n'.format(x = graph['Unnamed: 0'][ind], y = att))
        if graph['Unnamed: 0'][ind] not in statring_atts:
            #file.write('{x}0        ~ normal(0, 1);\n'.format(x = graph['Unnamed: 0'][ind]))
            data_to_file.append('{x}0        ~ normal(0, 1);\n'.format(x = graph['Unnamed: 0'][ind]))
        #file.write('\n')
        data_to_file.append('\n')
    #file.write('for(ind in 1:N){')
    if 0 in np.char.find(dataset_types,'float'):
        data_to_file.append('sigma_h_Sq ~ inv_gamma(1, 1);\n')
        data_to_file.append('\n')
    data_to_file.append('for(ind in 1:N){')
    graph_dic ={}
    for i in graph.columns[1:]:
        if np.where(graph[i])[0].shape[0]==0:
            graph_dic[i]=None
        else:
            graph_dic[i]= graph['Unnamed: 0'][np.where(graph[i])[0]].values
    time1 = time.time()
    while statring_atts.shape[0] != graph['Unnamed: 0'].shape[0]:
        if time.time() - time1>2:
            print('Not a DAG', file_num)
            not_a_DAG = True
            break
        for att in graph_dic.keys():
    #         print('------------')
    #         print(att)
    #         print(statring_atts)

            if att not in statring_atts:
                if 0 in  [1 if graph_dic[att][i] in statring_atts else 0 for i in range(graph_dic[att].shape[0])]:
                    #print('jump')
                    continue
                else:
                    string =''
                    for i in range(graph_dic[att].shape[0]):
                        string += '({y}{x} * {y}[ind])  +'.format(x=att, y=graph_dic[att][i])
                    #if att == 'hours-per-week':input()
                    if np.unique(data[att]).shape[0]==2:
                        string = '{x}[ind] ~ bernoulli_logit('.format(x = att) + string + ' {x}0'.format(x=att)+');'

                    elif 'float' in dataset_types[np.where(data.columns==att)[0][0]]:
                        string = '{x}[ind] ~ normal('.format(x = att) + string + ' {x}0'.format(x=att)+', sigma_h);' 
                    else:    
                        string = '{x}[ind] ~ poisson(exp('.format(x = att) + string + ' {x}0'.format(x=att)+'));'
                    statring_atts = np.append(statring_atts,att) 
                    #file.write(string+'\n')
                    data_to_file.append(string+'\n')
    #file.write('}\n')
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    #file.write('}\n')
    data_to_file.append('}\n')
    #file.write('\n')
    data_to_file.append('\n')
    if not_a_DAG == False:
        print('DAG ', file_num)
        with open('../'+dataset+'_Analysis/'+Algorithm+'/PP/'+dataset+'_'+Algorithm+'_DAG_{file_num}.stan'.format(file_num=file_num),'w') as file:
            file.writelines(data_to_file)