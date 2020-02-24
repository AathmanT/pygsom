from sklearn.metrics import confusion_matrix
import math

def evaluate(classifier,Y_test,y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_test.astype(int), y_pred).ravel()

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f_score = 2*precision*recall /(precision+recall)
    interVal= (tp+fn)*(tn+fp)
    g_mean = math.sqrt(tp*tn/interVal)
    AUC = (tp/(tp+fn)+tn/(tn+fp))/2

    print(classifier," finished executing")
    # print("\nClassifier: "+ classifier)
    # print("f_score: " + str(f_score))
    # print("g_mean: " + str(g_mean))
    # print("AUC value: "+ str(AUC) +"\n")

    return [classifier,f_score,g_mean,AUC]