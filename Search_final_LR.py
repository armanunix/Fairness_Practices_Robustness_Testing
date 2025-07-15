import sys
sys.path.append("./subjects/")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler
import pandas as pd   
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
import os
from statsmodels.stats.weightstats import ztest as ztest
import math
import time
import copy
import re
import csv
import argparse
import itertools
from sklearn.model_selection import train_test_split
from adf_utils.config import census, credit, bank, compas, meps21
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.compas import compas_data
from adf_data.meps21 import meps21_data
import xml_parser
import xml_parser_domains
from Timeout import timeout
import cloudpickle
from fairness_metrics import fair_metrics
#AI Fairness 360
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from mutation import mutate, clip_LR, clip_SVM, clip_TreeReg, clip_DT
from IPython.display import clear_output
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", help='The name of dataset: census, credit, bank ')
# parser.add_argument("--algorithm", help='The name of algorithm: logistic regression, SVM, Random Forest')
# parser.add_argument("--sensitive_index", help='The index for sensitive feature')
# parser.add_argument("--output", help='The name of output file', required=False)
# parser.add_argument("--time_out", help='Max. running time', default = 14400, required=False)
# parser.add_argument("--max_iter", help='The maximum number of iterations', default = 100000, required=False)
# parser.add_argument("--original_model", help='Whether consider the original (unmitigated) model', default = False, required=False)
# parser.add_argument("--save_model", help='Whether to save models', default = False, required=False)
# args = parser.parse_args()
dataset='compas'
algorithm ='logistic regression'
mit_alg = 'exponenciated gradient'
lib='Fairlearn'
sensitive_index = 9
time_out = 7200 * 4
original_model = True
save_model = False
output ='test12'
# parser.add_argument("--output", help='The name of output file', required=False)
# parser.add_argument("--time_out", help='Max. running time', default = 14400, required=False)
# parser.add_argument("--max_iter", help='The maximum number of iterations', default = 100000, required=False)
# parser.add_argument("--original_model", help='Whether consider the original (unmitigated) model', default = False, required=False)
# parser.add_argument("--save_model", help='Whether to save models', default = False, required=False)
def Binary_dataset(df, favorable_label, unfavorable_label,label_names, protected_attribute_names ):
    
    dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=df,
                                label_names=label_names,
                                protected_attribute_names=protected_attribute_names)
        
    return dataset

def check_for_fairness_ai360(dataset, y_pred, sensitive_name, unpriviliged_group, priviliged_group):
    """
    Returns metrics for entire dataset using AI Fairness 360
    """

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred

    classified_metric = ClassificationMetric(dataset, dataset_pred, 
                                             unprivileged_groups=unpriviliged_group, 
                                             privileged_groups=priviliged_group)
    metric_pred = BinaryLabelDatasetMetric(dataset_pred, 
                                           unprivileged_groups=unpriviliged_group, 
                                           privileged_groups=priviliged_group)    
    result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
             'disparate_impact': metric_pred.disparate_impact(),
             'equal_opportunity_difference': classified_metric.equal_opportunity_difference(),
             'accuracy': classified_metric.accuracy(),
             'precision': classified_metric.precision(),
             'recall': classified_metric.recall(),
             'false positive rate': classified_metric.false_positive_rate_difference(),
             'true positive rate': classified_metric.true_positive_rate_difference(),
             'selection rate': classified_metric.selection_rate(),
             'average_odds_difference':classified_metric.average_odds_difference(),
             'consistency':metric_pred.consistency(n_neighbors=5)}
        
    return result

#@timeout(time_out)#(int(args.time_out))
def test_cases(bin_dataset_train, bin_dataset_test,dataset, program_name, max_iter, sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, start_time, lib, original_model = False, save_model = False):
    time1 = time.time()
    df_org = pd.DataFrame()
    df_FL = pd.DataFrame()
    df_AI = pd.DataFrame()
    df_input = pd.DataFrame()

    program_name="LogisticRegressionMitigation_ExponentiatedGradient"
    #num_args = 0
#     input_program_tree_2 = 'ExponentiatedGradient_Params.xml'
#     num_args_2 = 4
    if lib == 'AI360':
        
        if(program_name == "LogisticRegressionMitigation_ExponentiatedGradient"):
            import Logistic_Regression_Mitigation_AI360
            mitigation_program = Logistic_Regression_Mitigation_AI360.LogisticRegressionMitigation_ExponentiatedGradient
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15  
        if(program_name == "LogisticRegressionMitigation_GridSearch"):
            import Logistic_Regression_Mitigation
            import Logistic_Regression_Mitigation_AI360
            input_program = Logistic_Regression_Mitigation.Mitigation_GridSearch
            input_program_ai360 = Logistic_Regression_Mitigation_AI360.LogisticRegressionMitigation_GridSearch
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
    else:    
        if(program_name == "LogisticRegressionMitigation_ExponentiatedGradient"):
            import Logistic_Regression_Mitigation
            import Logistic_Regression_Mitigation_AI360
            mitigation_program = Logistic_Regression_Mitigation.Mitigation_ExponentiatedGradient
            original_program = Logistic_Regression_Mitigation.logisticRegressionOriginal
            input_program_ai360 = Logistic_Regression_Mitigation_AI360.LogisticRegressionMitigation_ExponentiatedGradient
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
            input_program_tree_2 = 'ExponentiatedGradient_Params.xml'
            num_args_2 = 4
        elif(program_name == "Decision_Tree_Classifier_ExponentiatedGradient"):
            import Decision_Tree_Classifier_Mitigation
            mitigation_program = Decision_Tree_Classifier_Mitigation.Mitigation_ExponentiatedGradient
            input_program_tree = 'Decision_Tree_Classifier_Params.xml'
            num_args = 13
        elif(program_name == "TreeRegressorMitigation_ExponentiatedGradient"):
            import TreeRegressor_Mitigation
            mitigation_program = TreeRegressor_Mitigation.Mitigation_ExponentiatedGradient
            input_program_tree = 'TreeRegressor_Params.xml'
            num_args = 18
        elif(program_name == "SVM_Mitigation_ExponentiatedGradient"):
            import Logistic_Regression_Mitigation
            mitigation_program = Logistic_Regression_Mitigation.Mitigation_ExponentiatedGradient
            original_program = Logistic_Regression_Mitigation.SVMOriginal
            input_program_tree = 'SVM_Params.xml'
            num_args = 12
            input_program_tree_2 = 'ExponentiatedGradient_Params.xml'
            num_args_2 = 4
        elif(program_name == "LogisticRegressionMitigation_GridSearch"):
            import Logistic_Regression_Mitigation
            mitigation_program = Logistic_Regression_Mitigation.Mitigation_GridSearch
            original_program = Logistic_Regression_Mitigation.logisticRegressionOriginal
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
            input_program_tree_2 = 'GridSearch_Params.xml'#'ExponentiatedGradient_Params.xml'
            num_args_2 = 7
        
        elif(program_name == "LogisticRegressionMitigation_ThresholdOptimizer"):
            import Logistic_Regression_Mitigation
            mitigation_program = Logistic_Regression_Mitigation.Mitigation_ThresholdOptimizer
            original_program = Logistic_Regression_Mitigation.logisticRegressionOriginal
            input_program_tree = 'logistic_regression_Params.xml'
            num_args = 15
            input_program_tree_2 = 'ThresholdOptimizer_Params.xml'
            num_args_2 = 6
            
    arr_min_1, arr_max_1, arr_type_1, arr_default_1 = xml_parser_domains.xml_parser_domains(input_program_tree, num_args)
    arr_min_2, arr_max_2, arr_type_2, arr_default_2 = xml_parser_domains.xml_parser_domains(input_program_tree_2, num_args_2)
    arr_min, arr_max, arr_type, arr_default = arr_min_1 + arr_min_2, arr_max_1 + arr_max_2, arr_type_1 + arr_type_2, arr_default_1 + arr_default_2
    promising_inputs_TPR = []
    promising_inputs_FPR = []
    promising_inputs_AOD = []
    promising_inputs_EOD = []
    promising_metric_TPR = []
    promising_metric_FPR = []
    promising_metric_AOD = []
    promising_metric_EOD = []
    X_train = bin_dataset_train.convert_to_dataframe()[0].to_numpy()[:,:-1]
    y_train = bin_dataset_train.convert_to_dataframe()[0].to_numpy()[:,-1]
    X_test  = bin_dataset_test.convert_to_dataframe()[0].to_numpy()[:,:-1]
    y_test  = bin_dataset_test.convert_to_dataframe()[0].to_numpy()[:,-1]
    A_train = bin_dataset_train.convert_to_dataframe()[0][sensitive_name]
    A_test  = bin_dataset_test.convert_to_dataframe()[0][sensitive_name]

    
    high_diff_TPR = 0.0
    high_diff_FPR = 0.0
    low_diff_TPR = 1.0
    low_diff_FPR = 1.0
    default_acc = 0.0
    failed = 0
    highest_acc = 0.0
    highest_acc_inp = None
    AOD_diff_high = 0.0
    AOD_diff_low = 0.0
    EOD_diff_high = 0.0
    EOD_diff_low = 0.0
    AOD_diff  = 0.0
    
    if output == None:
        filename = "./Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_mutation_" + str(int(start_time)) + "_res.csv"
    elif output == "":
        filename = "./Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_mutation_" + str(int(start_time)) + "_res.csv"
    elif ".csv" in output:
        filename = "./Dataset/" + output#args.output
    else:
        filename = "./Dataset/" + output + ".csv"
    
    
    result = {}
    mu = [-1,0]
    sigma = [0,0]
    explore = True
    num_samples = 30
    explanation_EODs = {}
    best_dist = np.zeros((2,num_samples))
    for counter in range(max_iter):
        print('counter',counter)
        inp_orig  = mutate(arr_max_1, arr_min_1, arr_type_1, arr_default_1, promising_inputs_EOD, counter)
        inp_mut   = mutate(arr_max_2, arr_min_2, arr_type_2, arr_default_2, promising_inputs_EOD, counter)
        inp = inp_orig + inp_mut 

        if re.match(r'Logistic', program_name):
            arr_clip, features = clip_LR(inp,input_program_tree_2 ,num_args)
        elif re.match(r'Decision', program_name):
            arr_clip, features = clip_DT(inp,input_program_tree_2 , num_args)
        elif re.match(r'TreeRegressor', program_name):
            arr_clip, features = clip_TreeReg(inp,input_program_tree_2 , num_args)
        elif re.match(r'SVM', program_name):
            arr_clip, features = clip_SVM(inp,input_program_tree_2 , num_args)
        
        diff_EOD_list = np.zeros((num_samples))
        for test in range(num_samples):
            print('Test', test)
            if test == 0:
                res1, estimator, inp_valid1, score_org, pred_org = original_program(bin_dataset_train,bin_dataset_test,arr_clip, X_train, X_test, y_train, y_test, A_train, A_test, sensitive_param, program_name, dataset, "mutation", save_model, start_time, index=num_args)
                from sklearn.metrics import f1_score
            if not res1:
                print('training failed')
                failed += 1
                failed_flag = True
                break
            else:
                failed_flag = False

            res, LR, inp_valid, score_mit, pred_mit = mitigation_program(estimator, bin_dataset_train,bin_dataset_test,arr_clip, X_train, X_test, y_train, y_test, A_train, A_test, sensitive_param, program_name, dataset, "mutation", save_model, start_time, index=num_args)              

            if not res:
                print('mitigation failed')
                failed += 1
                failed_flag = True
                break
            else:
                failed_flag = False
                #print('f1 org, mit',f1_score(y_test,pred_org),f1_score(y_test,pred_mit))

            if counter == 0:
                default_acc = score_org

            if test == 0 :
                ai_result_A = check_for_fairness_ai360(bin_dataset_test, pred_org, sensitive_name, unpriviliged_group, priviliged_group)
            ai_result_B = check_for_fairness_ai360(bin_dataset_test, pred_mit, sensitive_name,  unpriviliged_group, priviliged_group)


            # A original model , B mitigated model

            EOD_A = ai_result_A['equal_opportunity_difference']
            EOD_B = ai_result_B['equal_opportunity_difference']

            AOD_A = ai_result_A['average_odds_difference']
            AOD_B = ai_result_B['average_odds_difference']

            TPR_A = ai_result_A['true positive rate']
            TPR_B = ai_result_B['true positive rate']
            FPR_A = ai_result_A['false positive rate']
            FPR_B = ai_result_B['false positive rate']

            # tolerate at most 2% of accuracy loss
            print('acc' ,score_org, score_mit )
            print('EOD',EOD_B , EOD_A)
            if (score_org < (default_acc - 0.02)) or (score_mit < score_org - 0.05):
                print('Acc low')
                acc_flag = True
                break
            else:
                acc_flag = False

            if(score_mit > highest_acc):
                highest_acc = score_mit
                highest_acc_inp = inp_valid


            diff_EOD = abs(EOD_B) - abs(EOD_A)
            print('diff_EOD',diff_EOD)
            diff_AOD = abs(AOD_B) - abs(AOD_A)
            diff_TPR = abs(TPR_B) - abs(TPR_A)
            diff_FPR = abs(FPR_B) - abs(FPR_A)
            diff_EOD_list[test] = diff_EOD
            if diff_EOD > mu[0] - sigma[0]:
                explore = True
                print('intresting')

            else:
                explore = False
                print('not intresting')
                break

        #For algorithm with randomness
        if (not failed_flag) and (not acc_flag) and num_samples != 1:
            df_org = pd.concat([df_org,pd.DataFrame.from_dict(ai_result_A, orient='index').T], sort=False) 
            df_FL = pd.concat([df_FL,pd.DataFrame.from_dict(ai_result_B, orient='index').T], sort=False) 
            #df_AI = pd.concat([df_AI,pd.DataFrame.from_dict(ai_result_B1, orient='index').T], sort=False) 
            df_input = pd.concat([df_input,pd.DataFrame(arr_clip).T], sort=False)
            explanation_EODs[counter] = diff_EOD_list
            if counter==0:

                mu[0] = np.mean(diff_EOD_list)
                sigma[0] = np.std(diff_EOD_list)
                best_dist[0] = diff_EOD_list
            else :
                if explore == True:
                    best_dist[1] = diff_EOD_list
                    mu[1] = np.mean(diff_EOD_list)
                    sigma[1] = np.std(diff_EOD_list)
                    #z_score = (mu[0]-mu[1])/math.sqrt((pow(sigma[0],2)/num_samples)+(pow(sigma[1],2)/num_samples))
#                             print(dif_list)
#                             print(z_score)
                    z_test = ztest(best_dist[0],best_dist[1],value=0)
                    z_score = z_test[0]
                    p_value = z_test[1]

                    print('Z score',z_score)
                    if z_score < -1.645:
                        print('Default distribution changed', diff_EOD)
                        mu[0] = mu[1]
                        sigma[0] = sigma[1]
                        best_dist[0] = best_dist[1]
                        promising_inputs_EOD.append(inp)
                        promising_metric_EOD.append([diff_EOD, score_mit])
                        EOD_diff_high = diff_EOD

        #For algorithm without randomness
        if (not failed_flag) and (not acc_flag) and num_samples == 1:
            df_org = pd.concat([df_org,pd.DataFrame.from_dict(ai_result_A, orient='index').T], sort=False) 
            df_FL = pd.concat([df_FL,pd.DataFrame.from_dict(ai_result_B, orient='index').T], sort=False) 
            #df_AI = pd.concat([df_AI,pd.DataFrame.from_dict(ai_result_B1, orient='index').T], sort=False) 
            df_input = pd.concat([df_input,pd.DataFrame(arr_clip).T], sort=False)
            explanation_EODs[counter] = diff_EOD_list
            if diff_EOD > EOD_diff_high:
                promising_inputs_EOD.append(inp)
                promising_metric_EOD.append([diff_EOD, score_mit])
                EOD_diff_high = diff_EOD
                print('Higher EOD diff achieved')

        if counter == 0:
            promising_inputs_TPR.append(inp)
            promising_inputs_FPR.append(inp)
            promising_inputs_AOD.append(inp)
            promising_inputs_EOD.append(inp)
#             promising_metric_TPR.append([diff_TPR, score_mit])
#             promising_metric_FPR.append([diff_FPR, score_mit])
#             promising_metric_AOD.append([diff_AOD, score_mit])
#             promising_metric_EOD.append([diff_EOD, score_mit])
#             high_diff_TPR = diff_TPR
#             high_diff_FPR = diff_FPR
#             low_diff_TPR = diff_TPR
#             low_diff_FPR = diff_FPR


        print("---------------------------------------------------------")                 
        clear_output(wait=False)
        if time.time() - time1 >= time_out:
            break
    
    if not os.path.exists('./Results/'):
        os.makedirs('./Results/')
    if not os.path.exists('./Results/' + str(program_name) + '/'):
        os.makedirs('./Results/' + str(program_name) + '/')
    if not os.path.exists('./Results/' + str(program_name) + '/'+ str(dataset) + '/'):
        os.makedirs('./Results/' + str(program_name) + '/'+ str(dataset) + '/')
    if not os.path.exists('./Results/' + str(program_name) + '/'+ str(dataset) + '/' + str(sensitive_name) + '/'):
        os.makedirs('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/')
    df_org.to_csv('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/' + 'original.csv')
    df_FL.to_csv('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/'+ 'Fairlearn.csv')
    #df_AI.to_csv('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/' + 'FAI360_0.csv')
    df_input.to_csv('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/' + 'Inputs.csv')
    np.save('./Results/' + str(program_name) + '/' + str(dataset) + '/' + str(sensitive_name) + '/' + 'explanation_EODs.npy', explanation_EODs)
    print("------------------END-----------------------------------")


if __name__ == '__main__':
    start_time = time.time()
    dataset = dataset#args.dataset
    algorithm = LogisticRegression#, Decision_Tree_Classifier, TreeRegressor, Discriminant_Analysis
    #algorithm = args.algorithm
    num_iteration =  100000#int(args.max_iter)

    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data,
           'meps21':meps21_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, 'meps21': meps21}

    # census (9 is for sex: 0 (men) vs 1 (female); 8 is for race: 0 (white) vs 4 (black))
    # credit ...
    # bank ...
#     sensitive_param = 2#int(args.sensitive_index)
#     sensitive_name = ""

    for dataset in ['bank','census']:#data_config.keys():
        if dataset == "census":
            sens_list =[8]
        elif dataset == "credit":
            sens_list =[9]
        elif dataset == "bank":
            sens_list =[1]
        elif dataset == "compas":
            sens_list =[1, 2, 3]
        elif dataset == "meps21":
            sens_list =[2, 10]
        for sensitive_param in sens_list:
            if dataset == "census" and sensitive_param == 9:
                sensitive_name = "sex"
                priviliged_group = [{sensitive_name: 1.0}]  #male
                unpriviliged_group = [{sensitive_name: 0.0}]  #female
                favorable_label  = 1.0
                unfavorable_label = 0.0

            if dataset == "census" and sensitive_param == 8:
                sensitive_name = "race"
                priviliged_group = [{sensitive_name: 4.0}]
                unpriviliged_group = [{sensitive_name: 0.0}]
                favorable_label  = 1.0
                unfavorable_label = 0.0

            if dataset == "credit" and sensitive_param == 9:
                sensitive_name = "sex"
                priviliged_group = [{sensitive_name: 1.0}]  #male
                unpriviliged_group = [{sensitive_name: 0.0}]  #female
                favorable_label  = 1.0
                unfavorable_label = 0.0

            if dataset == "bank" and sensitive_param == 1:  # with 3,5: 0.89; with 2,5: 0.84; with 4,5: 0.05; with 3,4: 0.6
                sensitive_name = "age"
                priviliged_group = [{sensitive_name: 5.0}]
                unpriviliged_group = [{sensitive_name: 3.0}]
                favorable_label  = 1.0
                unfavorable_label = 0.0

            if dataset == "compas" and sensitive_param == 1:  # sex
                sensitive_name = "sex"
                priviliged_group = [{sensitive_name: 0.0}]  #female
                unpriviliged_group = [{sensitive_name: 1.0}]  #male
                favorable_label  = 0.0
                unfavorable_label = 1.0

            if dataset == "compas" and sensitive_param == 2:  # age
                sensitive_name = "age"
                priviliged_group = [{sensitive_name: 2.0}] # greater than 45
                unpriviliged_group = [{sensitive_name: 0.0}] # under 25
                favorable_label  = 0.0
                unfavorable_label = 1.0

            if dataset == "compas" and sensitive_param == 3:  # race
                sensitive_name = "race"
                priviliged_group = [{sensitive_name: 1.0}] # Caucasian
                unpriviliged_group = [{sensitive_name: 0.0}] # non-Caucasian
                favorable_label  = 0.0
                unfavorable_label = 1.0

            if dataset == "meps21" and sensitive_param == 2:  # race
                sensitive_name = "race"
                priviliged_group = [{sensitive_name: 1.0}] # white
                unpriviliged_group = [{sensitive_name: 0.0}] # not white
                favorable_label  = 1.0
                unfavorable_label = 0.0 

            if dataset == "meps21" and sensitive_param == 10:  # sex
                sensitive_name = "sex"
                priviliged_group = [{sensitive_name: 1.0}] # male
                unpriviliged_group = [{sensitive_name: 0.0}] # female
                favorable_label  = 1.0
                unfavorable_label = 0.0


            X, Y, input_shape, nb_classes = data[dataset]()

            df = pd.DataFrame(X)
            df.columns = data_config[dataset]().feature_name
            scaler = StandardScaler()
            for col in df.columns:
                    if col==sensitive_name:
                        continue
                    elif len(df[col].unique()) > 10 :
                        df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1,1))
            A = df[sensitive_name].to_numpy()
            X = df.to_numpy()
            Y = np.argmax(Y, axis=1)

            df['label'] = Y

            X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, Y, A, random_state=0)

            df_test = pd.DataFrame(X_test)
            df_test.columns = data_config[dataset]().feature_name
            df_test['label'] = y_test        

            df_train = pd.DataFrame(X_train)
            df_train.columns = data_config[dataset]().feature_name
            df_train['label'] = y_train

            #original_model = True if args.original_model == "True" else False
            #save_model = True if args.save_model == "True" else False
        #     try:
            #sensitive_name = df_train.columns[sensitive_param - 1]

            bin_dataset_train = Binary_dataset(df_train, favorable_label, unfavorable_label, 
                                               ['label'],[sensitive_name])
            bin_dataset_test = Binary_dataset(df_test, favorable_label, unfavorable_label, 
                                               ['label'],[sensitive_name])
            #Check if Binary datasets are valid
            if bin_dataset_train.validate_dataset() == None :
                print('Train dataset is valid')
            else:
                print('Train dataset is not valid')
            if bin_dataset_test.validate_dataset() == None :
                print('Test dataset is valid')
            else:
                print('Test dataset is not valid')
            print(f'Start testing {str(dataset)} dataset with {str(sensitive_name)} as the protected attribute.')

            test_cases(bin_dataset_train, bin_dataset_test, dataset, algorithm, num_iteration, 
                       sensitive_param, unpriviliged_group, priviliged_group, sensitive_name, 
                       start_time, lib, original_model, save_model)
        #     except TimeoutError as error:
        #         print("Caught an error!" + str(error))
        #         print("--- %s seconds ---" % (time.time() - start_time))

        #     print("--- %s seconds ---" % (time.time() - start_time))
