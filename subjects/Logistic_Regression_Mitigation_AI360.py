import itertools
import time
import xml_parser

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import EqualizedOdds
import cloudpickle
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction as mitigator
from aif360.algorithms.inprocessing import GridSearchReduction as GridSearch

def LogisticRegressionMitigation_ExponentiatedGradient(bin_dataset_train,bin_dataset_test,arr, features, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
    try:

        arr_clf, arr_preds, arr_score = [], [], []
        estimator = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
        C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
        solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
        random_state=arr[11], class_weight = arr[10], verbose = arr[12],
        warm_start = arr[13], n_jobs=arr[14])
        
        clf = mitigator(estimator,
        constraints=EqualizedOdds(),eps = arr[15], max_iter = arr[16], nu=None, eta0 = arr[17],
        run_linprog_step = arr[18],drop_prot_attr=True)
        
        #clf_mit_fitted = clf.fit(X_train, y_train, sensitive_features=A_train)
        clf_mit_fitted = clf.fit(bin_dataset_train)
        preds = clf.predict(bin_dataset_test).labels.reshape(8141,).astype('int')
        score = np.sum(y_test == preds)/len(y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
        
        if original:
            arr_clf.append(clf)
            arr_score.append(score)
            arr_preds.append(preds)
            clf2 = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
                    C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
                    solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
                    random_state=arr[11], class_weight = arr[10], verbose = arr[12],
                    warm_start = arr[13], n_jobs=arr[14])
            arr_clf.append(clf2)
            clf2_fitted = clf2.fit(X_train, y_train)
            score2 = clf2.score(X_test, y_test)
            arr_score.append(score2)
            preds2 = clf2.predict(X_test)
            arr_preds.append(preds2)
            if save_model:
                with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}_original.pkl", "wb") as file:
                    cloudpickle.dump(clf2_fitted, file)

#        print("here1")
    except ValueError as ve:
#	pass
        print("here you go------------------------------------")
#         print(arr)
#         print(ve)
        return False, None, arr, None, None, features
    # except KeyError:
    #     # print("here3")
    #     return False
    if original:
        return True, arr_clf, arr, arr_score, arr_preds, features
    else:
        return True, clf, arr, score, preds, features
    
    

def LogisticRegressionMitigation_GridSearch(bin_dataset_train,bin_dataset_test,arr, features, X_train, X_test, y_train, y_test ,A_train, A_test, sensitive_param, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):

    try:
        
        arr_clf, arr_preds, arr_score = [], [], []
        
        
        
        estimator = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
        C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
        solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
        random_state=arr[11], class_weight = arr[10], verbose = arr[12],
        warm_start = arr[13], n_jobs=arr[14])
        
        clf = GridSearch(estimator, constraints=EqualizedOdds(), grid_size=arr[15], drop_prot_attr=False)
        
        clf_mit_fitted = clf.fit(bin_dataset_train)
        preds = clf.predict(bin_dataset_test).labels.reshape(8141,).astype('int')
        score = np.sum(y_test == preds)/len(y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
        
        if original:
            arr_clf.append(clf)
            arr_score.append(score)
            arr_preds.append(preds)
            clf2 = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
                    C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
                    solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
                    random_state=arr[11], class_weight = arr[10], verbose = arr[12],
                    warm_start = arr[13], n_jobs=arr[14])
            arr_clf.append(clf2)
            clf2_fitted = clf2.fit(X_train, y_train)
            score2 = clf2.score(X_test, y_test)
            arr_score.append(score2)
            preds2 = clf2.predict(X_test)
            arr_preds.append(preds2)
            if save_model:
                with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}_original.pkl", "wb") as file:
                    cloudpickle.dump(clf2_fitted, file)

#        print("here1")
    except ValueError as ve:
#	pass
        print("here you go------------------------------------")
#         print(arr)
#         print(ve)
        return False, None, arr, None, None, features
    # except KeyError:
    #     # print("here3")
    #     return False
    if original:
        return True, arr_clf, arr, arr_score, arr_preds, features
    else:
        return True, clf, arr, score, preds, features
