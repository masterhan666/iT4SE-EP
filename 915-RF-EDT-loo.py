from featureGenerator import *
from readToMatrix import *
import numpy as np
import re
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, roc_curve
import sklearn
from sklearn.ensemble import RandomForestClassifier

def getMatrix(dirname):
    pssmList = os.listdir(dirname)
    pssmList.sort(key=lambda x: eval(x[:]))
    m = len(pssmList)
    reMatrix = np.zeros((m, 1600))
    for i in range(m):
        matrix = readToMatrix(dirname + '/' + pssmList[i], 'pssm')
        matrix = autoNorm(matrix, 'pssm')
        reMatrix[i, :] = getEDT(matrix,4)
    print(reMatrix.shape)
    return reMatrix

def main():
    x1 = getMatrix("data/Train915/result/negative/pssm_profile_uniref50")
    x2 = getMatrix("data/Train915/result/positive/pssm_profile_uniref50")
    x = np.vstack((x1, x2))
    y = [-1 for i in range(x1.shape[0])]
    y.extend([1 for i in range(x2.shape[0])])
    y = np.array(y)
    #
    N=x.shape[1]
    print(int(sqrt(N).real), N // 5, int(log(N, 2).real), N // 3, N // 2, N // 4, N//10)
    param_grid = {'max_features': [int(sqrt(N).real), N // 5, int(log(N, 2).real), N // 3, N // 2, N // 4, N//10]}
    gs = GridSearchCV(RandomForestClassifier(n_estimators=1000,random_state=1), param_grid, cv=10)
    gs.fit(x, y)
    print(gs.best_estimator_)
    print(gs.best_score_)

    #
    clf = gs.best_estimator_
    loo = LeaveOneOut()
    score = cross_val_score(clf, x, y, cv=loo).mean()
    print("LOO:{}".format(score))
    #
    loo_probas_y = []  #
    loo_test_y = []  #
    loo_predict_y = []  #
    for train, test in loo.split(x):
        clf.fit(x[train], y[train])
        loo_predict_y.extend(clf.predict(x[test]))  #
        loo_probas_y.extend(clf.predict_proba(x[test]))  #
        loo_test_y.extend(y[test])  #
    loo_probas_y = np.array(loo_probas_y)
    loo_test_y = np.array(loo_test_y)
    print(loo_probas_y.shape)
    #np.savetxt("915-RFclassification-EDT-LOO-probas_y.csv", loo_probas_y, delimiter=",")
    #np.savetxt("915-RFclassification-EDT-LOO-test_y.csv", loo_test_y, delimiter=",")
    #
    confusion = sklearn.metrics.confusion_matrix(loo_test_y, loo_predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC:{}".format(roc_auc_score(loo_test_y, loo_probas_y[:, 1])))
    print("SP:{}".format(TN / (TN + FP)))
    print("SN:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE:{}".format(TP / (TP + FP)))
    print("MCC:{}".format(n))
    print("F-score:{}".format((2 * TP) / (2 * TP + FP + FN)))
    print("ACC:{}".format((TP + TN) / (TP + FP + TN + FN)))

    #
    test_x1 = getMatrix("data/Test850/result/negative/pssm_profile_uniref50")
    test_x2 = getMatrix("data/Test850/result/positive/pssm_profile_uniref50")
    test_x = np.vstack((test_x1, test_x2))
    test_y = [-1 for i in range(test_x1.shape[0])]
    test_y.extend([1 for i in range(test_x2.shape[0])])
    clf = gs.best_estimator_
    clf.fit(x, y)
    predict_y = clf.predict(test_x)
    print("IND：{}".format(accuracy_score(test_y, predict_y)))


main()