# Importing the libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from gsmote.comparison_testing.Evaluator import evaluate
import gsmote.preprocessing as pp
from gsmote import GSMOTE
import sys
sys.path.append('../../')

date_file = "../../data/echoli.csv".replace('\\','/')
X,y = pp.pre_process(date_file)

def linear_training(X,y):
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train,y_train = GSMOTE.OverSample(X_t,y_t)
                                                                                                                   
    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred  = np.where(y_predict>0.5,1,0)

    evaluate("Linear Regression",y_test,y_pred)


def gradient_boosting(X,y):
     X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

     X_train,y_train = GSMOTE.OverSample(X_t,y_t)

     # Fitting Gradient boosting
     gbc = GradientBoostingClassifier (n_estimators=100, learning_rate = 0.01, max_depth = 3)
     gbc.fit(X_train, y_train)

     # Predicting the Test set results
     y_predict = gbc.predict(X_test)
     y_pred  = np.where(y_predict.astype(int)>0.5,1,0)

     evaluate("Gradient Boosting",y_test,y_pred)

def KNN(X,y):
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=1 / 4, random_state=0)
    X_train, y_train = GSMOTE.OverSample(X_t, y_t)
    # X_train,y_train = X_t,y_t
    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test).astype(int)

    # find confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test.astype(int), y_pred)
    print(cm)

    evaluate("KNN",y_test, y_pred)


def decision_tree(X, y):
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train, y_train = GSMOTE.OverSample(X_t, y_t)

    # Fitting Simple Linear Regression to the Training set
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    evaluate("Decision Tree", y_test, y_pred)


linear_training(X,y)
gradient_boosting(X,y)
KNN(X,y)
decision_tree(X,y)


