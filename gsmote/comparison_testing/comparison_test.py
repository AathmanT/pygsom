# Importing the libraries
# import sys
# sys.path.append('/content/pygsom/')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from gsmote import GSMOTE
from gsmote.comparison_testing.Evaluator import evaluate
import gsmote.preprocessing as pp
from gsmote.comparison_testing.compare_visual import visualize_data as vs
import sys
import pandas as pd
import xgboost as xgb

sys.path.append('../../')

date_file = "../../data/adultmini.csv".replace('\\', '/')
# date_file = "content/pygsom/data/ecoli.csv".replace('\\', '/')

X, y = pp.preProcess(date_file)

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Visualize original data
vs(X_t, y_t, "Original data")

# oversample
X_train, y_train = GSMOTE.OverSample(X_t, y_t)
# visualize oversampled data
vs(X_train, y_train, "Oversampled ")

def linear_training():


    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Linear Regression", y_test, y_pred)


def gradient_boosting():

    # Fitting Gradient boosting
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate("Gradient Boosting", y_test, y_pred)


def XGBoost():

    # Fitting Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate("XGBoost", y_test, y_pred)


def KNN():

    # X_train,y_train = X_t,y_t
    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test).astype(int)

    return evaluate("KNN", y_test, y_pred)


def decision_tree():

    # Fitting Simple Linear Regression to the Training set
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Decision Tree", y_test, y_pred)


def MLPClassifier():

    # Fitting MLPClassifier to the Training set
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver='lbfgs', alpha=1e-5,
                        random_state=1)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test).astype(int)

    return evaluate("MLPClassifier", y_test, y_pred)

def GSOM_Classifier():

    # y_train = y_train.astype(int)
    y1 = np.copy(y_train)
    y = np.column_stack([y1, y_train])
    labels = ["Name", "label"]
    y = np.vstack((labels, y))
    frame = pd.DataFrame(y[1:, :], columns=y[0, :])


    from GSOM import GSOM

    gsom1 = GSOM(.83, X_train.shape[1], max_radius=4)

    gsom1.fit(X_train, 100, 50)
    gsom1.labelling_gsom(X_train, frame, "Name", "label")
    gsom1.finalize_gsom_label()

    y_pred = gsom1.predict_values(X_test, y_test)
    # print(y_pred)
    # print("complete")
    return evaluate("GSOM_Classifier",y_test,np.array(y_pred).astype(int))


performance1 = linear_training()
performance2 = gradient_boosting()
performance3 = XGBoost()
performance4 = KNN()
performance5 = decision_tree()
performance6 = MLPClassifier()
performance7 = GSOM_Classifier()


labels = ["Classifier", "f_score","g_mean","auc_value"]
values = [performance1,performance2,performance3,performance4,performance5,performance6,performance7]

scores = pd.DataFrame(values,columns=labels)
print(scores)