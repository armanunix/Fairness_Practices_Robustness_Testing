import itertools
import time
import xml_parser
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, mean_squared_error
import cloudpickle
from sklearn.utils import check_random_state

def SVMOriginal(arr, X_train, X_test, y_train, y_test):
    try:
        clf = SVC(C=arr[0], kernel=arr[1], degree=arr[2], gamma=arr[3], coef0=arr[4], shrinking=arr[5], probability=arr[6], tol=arr[7], cache_size=arr[8], class_weight=arr[9], verbose=arr[10], max_iter=arr[11], decision_function_shape=arr[12], break_ties=arr[13], random_state=arr[14])

        clf_fitted = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        pred = clf.predict(X_test)


    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, arr, None, None

    return True, arr, score, pred 

#---------------------------------
def TreeRegressOriginal(bin_dataset_train,bin_dataset_test,arr, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, index = None):
    try:
        
#         ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples', 'selection_rule', 'constraint_weight', 'grid_size', 'grid_limit', 'grid_offset', 'grid', 'sample_weight_name']
        clf = RandomForestClassifier(n_estimators=arr[0], criterion=arr[1],
                max_depth=arr[2], min_samples_split=arr[3], min_samples_leaf=arr[4],
                min_weight_fraction_leaf=arr[5], max_features=arr[6],
                max_leaf_nodes=arr[7], min_impurity_decrease=arr[8],
                bootstrap=arr[9],oob_score=arr[10], n_jobs = arr[11], 
                random_state=arr[12], verbose = arr[13], warm_start=arr[14],
                class_weight=arr[15], ccp_alpha = arr[16], max_samples = arr[17])

        clf_fitted = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #pred = list(map(trans, pred))
        score = accuracy_score(pred, y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}_original.pkl", "wb") as file:
                cloudpickle.dump(clf_fitted, file)
    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve) 
        return False, None, arr, None, None
    

    return True, clf, arr, score, pred

#------------------------------
def logisticRegressionOriginal(arr, X_train, X_test, y_train, y_test):
    try:
        
        clf2 = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
                C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
                solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
                random_state=arr[11], class_weight = arr[10], verbose = arr[12],
                warm_start = arr[13], n_jobs=arr[14])

        clf2_fitted = clf2.fit(X_train, y_train)
        score = clf2.score(X_test, y_test)
        pred = clf2.predict(X_test)
        
    except ValueError as ve:
        print("here you go------------------------------------")
        print(ve)
        return False,clf2, arr, None, None
        
    return True,clf2, arr, score, pred 
def lassoRegressionOriginal(arr, X_train, X_test, y_train, y_test):
    try:
        
        clf2 = Lasso(alpha=arr[0], fit_intercept=arr[1], precompute=arr[2], copy_X=arr[3], max_iter=arr[4], 
                      tol=arr[5], warm_start=arr[6], positive=arr[7], random_state=arr[8], selection=arr[9])

        clf2_fitted = clf2.fit(X_train, y_train)
        
        #score = clf2.score(X_test, y_test)
        pred = clf2.predict(X_test)
        score = mean_squared_error(y_test,pred)

    except ValueError as ve:
        print('failed input',arr) 
        print("here you go------------------------------------")
        print(arr)
        print(ve) 
        return False, arr, None, None
        
    return True, arr, score, pred

def Mitigation_ExponentiatedGradient(estimator, bin_dataset_train,bin_dataset_test,arr, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, save_model=False, index = None):

    try:
        clf = ExponentiatedGradient(estimator,
        constraints=EqualizedOdds(),eps = arr[index], max_iter = arr[index+1], nu=None, eta0 = arr[index+2],
        run_linprog_step = arr[index+3])
        clf_mit_fitted = clf.fit(X_train, y_train, sensitive_features=A_train)
        #preds = clf.predict(X_test)
        preds_prob = clf._pmf_predict(X_test)
        random_state = check_random_state(None)
        preds = np.array([(preds_prob[:, 1] >= random_state.rand(len(preds_prob[:, 1]))) * 1 for i in range(30)])
        scores = np.array([ np.sum(y_test == preds[i])/len(y_test) for i in range(preds.shape[0])])
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
        


    except ValueError as ve:

        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None
    
    return True, clf, arr, scores, preds

def Mitigation_GridSearch(estimator, bin_dataset_train,bin_dataset_test,arr, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
    try:
        
        clf = GridSearch(estimator,
        constraints=EqualizedOdds(), selection_rule= arr[index], 
                        constraint_weight = arr[index+1], grid_size=arr[index+2], grid_limit =arr[index+3], 
                         grid_offset = arr[index+4], 
                        grid = arr[index+5], sample_weight_name = arr[index+6])
        clf_mit_fitted = clf.fit(X_train, y_train, sensitive_features=A_train)
        preds = clf.predict(X_test)
        score = np.sum(y_test == preds)/len(y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
        


    except ValueError as ve:

        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None
    
    return True, clf, arr, score, preds

def Mitigation_ThresholdOptimizer(estimator, bin_dataset_train,bin_dataset_test,arr, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
    try:
        un_estimator = estimator.fit(X_train, y_train)
        clf = ThresholdOptimizer(estimator=un_estimator,
        constraints=arr[index], objective = arr[index+1], grid_size = arr[index+2], flip = arr[index+3],
                                prefit = arr[index+4], predict_method = arr[index+5])
        clf_mit_fitted = clf.fit(X_train, y_train, sensitive_features=A_train)
        preds = clf.predict(X_test, sensitive_features=A_test)
        score = np.sum(y_test == preds)/len(y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
        


    except ValueError as ve:

        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None
    
    return True, clf, arr, score, preds
