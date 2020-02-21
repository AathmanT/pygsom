from collections import Counter
from gsmote import GeometricSMOTE
import sys
sys.path.append('../../')

def OverSample(X,y):
    print('Original dataset shape %s' % Counter(y))
    gsmote = GeometricSMOTE(random_state=1)
    X_res, y_res = gsmote.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))




    return X_res,y_res
