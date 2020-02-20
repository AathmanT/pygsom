# Importing the libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from gsmote.comparison_testing.Evaluator import evaluate
import gsmote.preprocessing as pp
from gsmote import GSMOTE as gs
import sys
sys.path.append('../../')

date_file = "../../data/adultmini.csv".replace('\\','/')
X,y = pp.preProcess(date_file)

def linear_training(X,y):
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train,y_train = X_t,y_t
                                                                                                                   
    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred  = np.where(y_predict>0.5,1,0)
    print(y_pred)

    evaluate(y_test,y_pred)

linear_training(X,y)



