from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

def preProcess(filename):
    # X, y = make_classification(n_classes=2, class_sep=2,
    # weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    # n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    df = pd.read_csv(filename)
    print(df)
    # Loading of Selected Features into X
    X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]].values

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [10])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [26])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [32])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [46])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [51])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    # Loading of the Label into y
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
    y1 = np.array(columnTransformer.fit_transform(df), dtype=np.str)
    y = y1[:, 1]
    y = np.where(y == "0.0", "0", y)
    y = np.where(y == "1.0", "1", y)
    # new = np.copy(y)
    # data = np.column_stack([X,y,new])
    # labels = []
    # for i in range(0,55):
    #     labels.append("w"+str(i))
    # labels = labels + ["Name","label"]
    # frame = pd.DataFrame(data,columns=labels)
    # export_csv = frame.to_csv(r'C:\Users\User\PycharmProjects\FYP\pygsom\data\export_dataframe.csv', index=None,
    #                        header=True)  # Don't forget to add '.csv' at the end of the path

    return X, y


def pre_process(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    y = df.iloc[:,-1].values
    return X,y

