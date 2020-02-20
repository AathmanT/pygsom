
# Importing the libraries
import numpy as np

import sys

from xlwt.antlr import ifelse

from gsmote import GSMOTE
from gsmote import preprocessing as pp
from gsmote.comparison_testing import Evaluator
sys.path.append('../../')

data_filename = "../../data/adult.csv".replace('\\', '/')
X,y = pp.preProcess(data_filename)

from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

X_train,y_train = GSMOTE.OverSample(X_t,y_t)
# X_train,y_train = X_t,y_t

# Fitting Simple Linear Regression to the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
prob_pred = regressor.predict(X_test)
y_pred = np.where(prob_pred>0.5, 1, 0).astype(int)

#find confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype(int), y_pred)
print(cm)

Evaluator.evaluate(y_test, y_pred)

# # Visualising the Training set results
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
#
# # Visualising the Test set results
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()