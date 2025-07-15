import pandas as pd
from itertools import product
data = pd.read_csv('./subjects/datasets/Bank')
dataset='Bank'
#data = pd.read_csv('./subjects/datasets/adult_org-Copy1.csv')
#dataset='Adult'
#data = pd.read_csv('./subjects/datasets/compas-Copy1')
#dataset='Compas'
#data = pd.read_csv('./subjects/datasets/heart_processed_1')
#dataset='Heart'
#data = pd.read_csv('./subjects/datasets/law.csv')
#dataset='Law'
#data = pd.read_csv('./subjects/datasets/students-processed_2')
#dataset='Student'
algorithm = 'pc'
#algorithm = 'ges'
#algorithm = 'simy'

if algorithm == 'pc':
    dag = pd.read_csv('./'+dataset+'_Analysis/'+algorithm+'/DAGs/'+dataset+'_'+algorithm+'.csv')
else:
    dag = pd.read_csv('./'+dataset+'_Analysis/'+algorithm+'/DAGs/'+dataset+'_'+algorithm+'.csv').astype(int)
dag = dag[dag.columns[1:]]
dag.columns = data.columns
dag.index = data.columns
if algorithm != 'pc':
    dag = dag.T

#dag.to_csv('../ges/Bank_ges_main.csv',index=False)
bidir_nodes =[]
for edge1 in dag.columns:
    for edge2 in dag.columns:
        if dag.loc[edge1,edge2] == 1 and dag.loc[edge2,edge1]==1 :
            if [edge1,edge2] not in bidir_nodes and  [edge2,edge1] not in bidir_nodes:
                bidir_nodes.append([edge1,edge2])
                
if len(bidir_nodes)!=0:                
    k=0
    for state in list(product([0, 1], repeat=len(bidir_nodes))):
        new_dag = dag.copy()
        for edges in bidir_nodes:
            new_dag.loc[edges[0],edges[1]]=0
            new_dag.loc[edges[1],edges[0]]=0
        for edg in range(len(state)):
            if state[edg]== 0 :

                #print('{edge1}->{edge2}'.format(edge1=bidir_nodes[edg][0],edge2=bidir_nodes[edg][1]))
                new_dag.loc[bidir_nodes[edg][0],bidir_nodes[edg][1]]=1


            else :
                #print('{edge1}<-{edge2}'.format(edge1=bidir_nodes[edg][0],edge2=bidir_nodes[edg][1]))
                new_dag.loc[bidir_nodes[edg][1],bidir_nodes[edg][0]]=1

        k += 1
        new_dag.to_csv('./'+dataset+'_Analysis/'+algorithm+'/DAGs/'+dataset+'_'+algorithm+'_DAG_{k}.csv'.format(k=k))
else:
    dag.to_csv('./'+dataset+'_Analysis/'+algorithm+'/DAGs/'+dataset+'_'+algorithm+'_DAG_1.csv')    