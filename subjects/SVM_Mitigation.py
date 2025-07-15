import itertools
import time
import xml_parser

import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
import cloudpickle

def SVM_Mitigation_ExponentiatedGradient(arr, features, X_train, X_test, y_train, y_test, A_train, A_test, sensitive_param = None, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
#     arr_1, features_1 = xml_parser.xml_parser('SVM_Params.xml',inp[:index])
#     arr_2, features_2 = xml_parser.xml_parser('ExponentiatedGradient_Params.xml',inp[index:])

#     arr, features = arr_1 + arr_2, features_1 + features_2

    try:
#         # domain-specific constraints
#         if(arr[8] == 'none'):
#             arr[8] = None
#         # The combination of penalty='l1' and loss='hinge' is not supported, Parameters: penalty='l1', loss='hinge', dual=False
#         if(arr[0] == 'l1' and arr[1] == 'hinge'):
#             arr[2] = True
#             arr[0] = 'l2'
#         if(arr[0] == 'l2' and arr[1] == 'hinge'):
#             arr[2] = True
#         if(arr[0] == 'l1' and arr[1] == 'squared_hinge'):
#             arr[2] = False
#         arr[5] = 'ovr'
#         arr[9] = 2019
#         arr[10] = 0
        arr_clf, arr_preds, arr_score = [], [], []
        clf = ExponentiatedGradient(LinearSVC(penalty = arr[0], loss = arr[1], dual = arr[2], tol = arr[3], C = arr[4],
                        multi_class = arr[5], fit_intercept = arr[6], intercept_scaling = arr[7],
                        class_weight = arr[8], verbose=arr[10], random_state=arr[9], max_iter=arr[11]),
                        constraints=EqualizedOdds(), eps = arr[12], max_iter = arr[13],
                        eta0 = arr[14], run_linprog_step = arr[15])
        clf_mit_fitted = clf.fit(X_train, y_train, sensitive_features=A_train)
        preds = clf.predict(X_test)
        score = np.sum(y_test == preds)/len(y_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)

        if original:
            arr_clf.append(clf)
            arr_score.append(score)
            arr_preds.append(preds)
            clf2 = LinearSVC(penalty = arr[0], loss = arr[1], dual = arr[2], tol = arr[3], C = arr[4],
                        multi_class = arr[5], fit_intercept = arr[6], intercept_scaling = arr[7],
                        class_weight = arr[8], verbose=arr[10], random_state=arr[9], max_iter=arr[11])
            arr_clf.append(clf2)
            clf2_fitted = clf2.fit(X_train, y_train)
            score2 = clf2.score(X_test, y_test)
            arr_score.append(score2)
            preds2 = clf2.predict(X_test)
            arr_preds.append(preds2)
            if save_model:
                with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}_original.pkl", "wb") as file:
                    cloudpickle.dump(clf2_fitted, file)

    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features
    print(arr)

    if original:
        return True, arr_clf, arr, arr_score, arr_preds, features
    else:
        return True, clf, arr, score, preds, features
