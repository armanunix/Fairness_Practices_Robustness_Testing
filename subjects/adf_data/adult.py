import numpy as np
import pandas as pd
import sys
sys.path.append("../")

raw_data = pd.read_csv('./subjects/datasets/adult_pp.csv',header='infer')

def adult_data():
    """
    Prepare the data of dataset Law School
    :return: X, Y, input shape and number of classes
    """
    X = np.array(raw_data.to_numpy()[:,:-1])
    Y = np.array(raw_data['label']).reshape(-1,1)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    return X, Y, input_shape, nb_classes