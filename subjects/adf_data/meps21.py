import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import MEPSDataset21
cd = MEPSDataset21()
df = cd.convert_to_dataframe()[0]
le = LabelEncoder()
df = df.rename(columns={'AGE':'age','RACE':'race','SEX=1':'sex'})
df.drop(columns=['SEX=2'], inplace=True)
df['age'] = pd.cut(df['age'],9, labels=[i for i in range(1,10)])

def meps21_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = np.array(df.to_numpy()[:,:-1], dtype=int)
    Y = np.array(cd.labels, dtype=int)
    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    return X, Y, input_shape, nb_classes

