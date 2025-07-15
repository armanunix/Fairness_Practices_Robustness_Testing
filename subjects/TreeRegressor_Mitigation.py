from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import xml_parser
from sklearn.datasets import make_regression
import random
from sklearn.metrics import accuracy_score
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
import cloudpickle

def trans(x):
    if(x >= 0.5):
        return 1
    else:
        return 0

def TreeRegressMitigation_ExponentiatedGradient(arr, features, X_train, X_test, y_train, y_test,A_train, A_test, sensitive_param = None, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
#     arr_1, features_1 = xml_parser.xml_parser('TreeRegressor_Params.xml',inp[:index])

#     arr_2, features_2 = xml_parser.xml_parser('ExponentiatedGradient_Params.xml',inp[index:])

#     arr, features = arr_1 + arr_2, features_1 + features_2
    
#     # mae is much slower than mse
#     if(arr[0] == 'mae'):
#         rand = np.random.random()
#         if(rand < 0.98):
#             arr[0] = 'mse'

#     # value for max depth
#     if(arr[1] == 'None'):
#         arr[1] = None
#     else:
#         arr[1] = random.randint(5, 20)

#     # value for max_features
#     if arr[5] == 'None':
#         arr[5] = None

#     if arr[6] == 'None':
#         arr[6] = None
#     arr[7] = 0.0
#     arr[8] = True

#     # if(arr[13] == 'None'):
#     arr[13] = None
#     arr[14] = 2019
#     arr[15] = 0
#     arr[16] = None
#     arr[17] = None

    arr_clf, arr_preds, arr_score = [], [], []

    try:
        random_forest = ExponentiatedGradient(RandomForestRegressor(n_estimators=arr[11], criterion=arr[0],
            max_depth=arr[1], min_samples_split=arr[2], min_samples_leaf=arr[3],
            min_weight_fraction_leaf=arr[4], max_features=arr[5],
            max_leaf_nodes=arr[6], min_impurity_decrease=arr[7],
            bootstrap=arr[8],oob_score=arr[9], warm_start=arr[10], ccp_alpha = arr[12],
            max_samples = arr[13], random_state = arr[14], verbose = arr[15], n_jobs = arr[16],
            min_impurity_split = arr[17]), constraints=EqualizedOdds(), eps = arr[18],
            max_iter = arr[19], eta0 = arr[20], run_linprog_step = arr[21])
    except ValueError as VE:
        print("error2: " + str(VE))
        return False, None, arr, None, None, features

    try:
        clf_mit_fitted = random_forest.fit(X_train, y_train, sensitive_features=A_train)
        x_pred = random_forest.predict(X_test)
        x_pred = list(map(trans, x_pred))
        score = accuracy_score(x_pred, y_test)
        preds = random_forest.predict(X_test)
        if save_model:
            with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}.pkl", "wb") as file:
                cloudpickle.dump(clf_mit_fitted, file)
    except ValueError as VE:
        print("error3: " + str(VE))
        return False, None, arr, None, None, features

    preds = list(map(trans, preds))
    if original:
        arr_clf.append(random_forest)
        arr_score.append(score)
        arr_preds.append(preds)

        try:
            random_forest = RandomForestRegressor(n_estimators=arr[11], criterion=arr[0],
                    max_depth=arr[1], min_samples_split=arr[2], min_samples_leaf=arr[3],
                    min_weight_fraction_leaf=arr[4], max_features=arr[5],
                    max_leaf_nodes=arr[6], min_impurity_decrease=arr[7],
                    bootstrap=arr[8],oob_score=arr[9], warm_start=arr[10], ccp_alpha = arr[12],
                    max_samples = arr[13], random_state = arr[14], verbose = arr[15], n_jobs = arr[16],
                    min_impurity_split = arr[17])
            arr_clf.append(random_forest)
        except ValueError as VE:
            print("error2: " + str(VE))
            return False, None, arr, None, None, features

        try:
            clf_fitted = random_forest.fit(X_train, y_train)
            x_pred = random_forest.predict(X_test)
            x_pred = list(map(trans, x_pred))
            score = accuracy_score(x_pred, y_test)
            arr_score.append(score)
            preds = random_forest.predict(X_test)
            preds = list(map(trans, preds))
            arr_preds.append(preds)
            if save_model:
                with open(f"./trained_models/{program_name}_{dataset_name}_{sensitive_param}_{search_name}_{str(int(start_time))}_original.pkl", "wb") as file:
                    cloudpickle.dump(clf_fitted, file)
        except ValueError as VE:
            print("error3: " + str(VE))
            return False, None, arr, None, None, features

    print(arr)

    if original:
        return True, arr_clf, arr, arr_score, arr_preds, features
    else:
        return True, random_forest, arr, score, preds, features
