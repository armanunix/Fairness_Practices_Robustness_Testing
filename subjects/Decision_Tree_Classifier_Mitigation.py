import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xml_parser
import random
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
import cloudpickle

def DecisionTreeMitigation_ExponentiatedGradient(arr, features, X_train, X_test, y_train, y_test,A_train, A_test, sensitive_param = None, program_name = "", dataset_name = "", search_name = "", save_model=False, start_time = 0, original = False, index = None):
#     arr_1, features_1 = xml_parser.xml_parser('Decision_Tree_Classifier_Params.xml',inp[:index])

#     arr_2, features_2 = xml_parser.xml_parser('ExponentiatedGradient_Params.xml',inp[index:])

#     arr, features = arr_1 + arr_2, features_1 + features_2

#     if(arr[2] == 'None'):
#         arr[2] = None
#     else:
#         if np.random.random() < 0.25:
#             arr[2] = 4
#         else:
#             arr[2] = random.randint(3, 20)

#     if arr[6] == 'None':
#         arr[6] = None

#     arr[7] = 2019

#     arr[8] = None

#     arr[9] = 0.0

#     arr[10] = 0.0

#     arr[11] = None

    

    try:
        arr_clf, arr_preds, arr_score = [], [], []
        clf = ExponentiatedGradient(DecisionTreeClassifier(criterion=arr[0], splitter=arr[1], max_depth=arr[2],
                min_samples_split=arr[3], min_samples_leaf=arr[4], min_weight_fraction_leaf=arr[5],
                max_features=arr[6], random_state=arr[7], max_leaf_nodes=arr[8],
                min_impurity_decrease=arr[9], class_weight=arr[11],
                ccp_alpha = arr[12]), constraints=EqualizedOdds(), eps = arr[13], max_iter = arr[14],
                eta0 = arr[15], run_linprog_step = arr[16])
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
            clf2 = DecisionTreeClassifier(criterion=arr[0], splitter=arr[1], max_depth=arr[2],
                min_samples_split=arr[3], min_samples_leaf=arr[4], min_weight_fraction_leaf=arr[5],
                max_features=arr[6], random_state=arr[7], max_leaf_nodes=arr[8],
                min_impurity_decrease=arr[9], class_weight=arr[11],
                ccp_alpha = arr[12])
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
#	pass
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, None, None, None, None

    print(arr)
    if original:
        return True, arr_clf, arr, arr_score, arr_preds, features
    else:
        return True, clf, arr, score, preds, features
