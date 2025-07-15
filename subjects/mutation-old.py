
import numpy as np
from scipy.stats import randint
import xml_parser

def mutate(arr_max, arr_min, arr_type, arr_default,  promising_inputs, counter):
    inp = []
    if counter == 0 :

        for i in range(len(arr_min)):
            if(arr_type[i] == 'bool'):
                inp.append(int(arr_default[i]))
            elif(arr_type[i] == 'int'):
                inp.append(int(arr_default[i]))
            elif(arr_type[i] == 'float'):
                inp.append(float(arr_default[i]))
    else:
        rnd = np.random.random()
        if ((rnd < 0.05 and counter > 100) or (rnd < 0.5 and counter < 100)) :
            for i in range(len(arr_min)):
                if(arr_type[i] == 'bool'):
                    inp.append(randint.rvs(0,2))
                elif(arr_type[i] == 'int'):
                    minVal = int(arr_min[i])
                    maxVal = int(arr_max[i])
                    inp.append(np.random.randint(minVal,maxVal+1))
                elif(arr_type[i] == 'float'):
                    minVal = float(arr_min[i])
                    maxVal = float(arr_max[i])
                    inp.append(np.random.uniform(minVal,maxVal+0.00001))
        else:
            # if rnd < 0.9:
            inp = promising_inputs[-1][-len(arr_max):]
            index = np.random.randint(0,len(arr_min)-1)
            if(arr_type[index] == 'bool'):
                inp[index] = 1 - inp[index]
            elif(arr_type[index] == 'int'):
                minVal = int(arr_min[index])
                maxVal = int(arr_max[index])
                rnd = np.random.random()
                if rnd < 0.4:
                    newVal = np.random.randint(minVal,maxVal+1)
                    trail = 0

                    while newVal == inp[index] and trail < 3:
                        newVal = np.random.randint(minVal,maxVal+1)
                        trail += 1
                elif rnd < 0.7:
                    newVal = inp[index] + 1
                else:
                    newVal = inp[index] - 1
                inp[index] = newVal
            elif(arr_type[index] == 'float'):
                minVal = float(arr_min[index])
                maxVal = float(arr_max[index])
                rnd = np.random.random()
                if rnd < 0.5:
                    inp[index] = np.random.uniform(minVal,maxVal+0.000001)
                elif rnd < 0.75:
                    newVal = inp[index] + abs(maxVal-minVal)/100
                    inp[index] = newVal
                else:
                    newVal = inp[index] - abs(maxVal-minVal)/100
                    inp[index] = newVal
                
                
    return inp

def clip_LR(inp, input_tree, index=None):
    
    arr_1, features_1 = xml_parser.xml_parser('logistic_regression_Params.xml',inp[:index])

    arr_2, features_2 = xml_parser.xml_parser(input_tree,inp[index:])
    
    arr, features = arr_1 + arr_2, features_1 + features_2
         # domain-specific constraints
    if (arr[0] == 'lbfgs' and arr[2] == True):
        arr[2] = False
    if (arr[0] == 'lbfgs' and arr[1] == "l1") or (arr[0] == 'lbfgs' and arr[1] == "elasticnet"):
        arr[1] = "l2"
    if (arr[0] == 'newton-cg' and arr[2] == True):
        arr[2] = False
    if (arr[0] == 'newton-cg' and arr[1] == "l1") or (arr[0] == 'newton-cg' and arr[1] == "elasticnet"):
        arr[1] = "l2"
    if (arr[0] == 'sag' and arr[2] == True):
        arr[2] = False
    if (arr[0] == 'sag' and arr[1] == "l1") or (arr[0] == 'sag' and arr[1] == "elasticnet"):
        arr[1] = "l2"
    if (arr[0] == 'liblinear' and arr[1] == "elasticnet"):
        arr[1] = "l1"
    if (arr[0] == 'lbfgs' and arr[1] == "elasticnet"):
        arr[1] = "l1"
    if (arr[0] == 'saga' and arr[2] == True):
        arr[2] = False
    if (arr[1] == "l1" and arr[2] == True):
        arr[2] = False
    if (arr[0] == 'liblinear' and arr[8] == 'multinomial'):
        arr[8] = 'ovr'
    if (arr[0] == 'liblinear' and arr[1] == 'none'):
        arr[1] = 'l1'
    if(arr[1] == "none"):
        arr[4] = 1.0
    if(arr[1] != "elasticnet"):
        arr[9] = None
        
    else:
        arr[9] = np.random.random()
    arr[10] = None
    arr[11] = 2019
    arr[12] = 0
    arr[14] = None
    return arr, features 

def clip_DT(inp, input_tree, index=None):
    
    arr_1, features_1 = xml_parser.xml_parser('Decision_Tree_Classifier_Params.xml',inp[:index])
    arr_2, features_2 = xml_parser.xml_parser(input_tree,inp[index:])
    arr, features = arr_1 + arr_2, features_1 + features_2

    if(arr[2] == 'None'):
        arr[2] = None
    else:
        if np.random.random() < 0.25:
            arr[2] = 4
        else:
            arr[2] = np.random.randint(3, 20)

    if arr[6] == 'None':
        arr[6] = None

    arr[7] = 2019

    arr[8] = None

    arr[9] = 0.0

    arr[10] = 0.0

    arr[11] = None
        
    return arr, features

def clip_SVM(inp, input_tree, index=None):
    
    arr_1, features_1 = xml_parser.xml_parser('SVM_Params.xml',inp[:index])
    arr_2, features_2 = xml_parser.xml_parser(input_tree,inp[index:])
    arr, features = arr_1 + arr_2, features_1 + features_2

    
    # domain-specific constraints
    if(arr[8] == 'none'):
        arr[8] = None
    # The combination of penalty='l1' and loss='hinge' is not supported, Parameters: penalty='l1', loss='hinge', dual=False
    if(arr[0] == 'l1' and arr[1] == 'hinge'):
        arr[2] = True
        arr[0] = 'l2'
    if(arr[0] == 'l2' and arr[1] == 'hinge'):
        arr[2] = True
    if(arr[0] == 'l1' and arr[1] == 'squared_hinge'):
        arr[2] = False
    arr[5] = 'ovr'
    arr[9] = 2019
    arr[10] = 0
        
    return arr, features   



def clip_TreeReg(inp, input_tree, index=None):
    
    arr_1, features_1 = xml_parser.xml_parser('TreeRegressor_Params.xml',inp[:index])
    arr_2, features_2 = xml_parser.xml_parser(input_tree,inp[index:])
    arr, features = arr_1 + arr_2, features_1 + features_2
    
    # mae is much slower than mse
    if(arr[0] == 'mae'):
        rand = np.random.random()
        if(rand < 0.98):
            arr[0] = 'mse'

    # value for max depth
    if(arr[1] == 'None'):
        arr[1] = None
    else:
        arr[1] = random.randint(5, 20)

    # value for max_features
    if arr[5] == 'None':
        arr[5] = None

    if arr[6] == 'None':
        arr[6] = None
    arr[7] = 0.0
    arr[8] = True

    # if(arr[13] == 'None'):
    arr[13] = None
    arr[14] = 2019
    arr[15] = 0
    arr[16] = None
    arr[17] = None

        
    return arr, features
            
            