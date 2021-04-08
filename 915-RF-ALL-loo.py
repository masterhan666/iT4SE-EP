from featureGenerator import *
from readToMatrix import *
import numpy as np
import re
import os
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, roc_curve
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sys
#
def autoNormTenFactor(matrix):
    for i in range(matrix.shape[1]):#lie
        mini=min(matrix[:,i])
        maxi=max(matrix[:,i])
        for j in range(matrix.shape[0]):#
            matrix[j][i]=(matrix[j,i]-mini)/(maxi-mini)
    return matrix
#
def Norm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    m=dataSet.shape[0]
    dataSet=dataSet-np.tile(minVals,(m,1))
    dataSet=dataSet/np.tile(ranges,(m,1))
    return dataSet

def getMatrix(dirname):
    pssmList = os.listdir(dirname)
    pssmList.sort(key=lambda x: eval(x[:]))
    m = len(pssmList)
    reMatrix = np.zeros((m, 2060+400*eval(sys.argv[1])))
    for i in range(m):
        matrix2 = readToMatrix(dirname + '/' + pssmList[i], 'psfm')
        matrix2 = autoNorm(matrix2, 'psfm')
        matrix = readToMatrix(dirname + '/' + pssmList[i], 'pssm')
        matrix = autoNorm(matrix, 'pssm')
        binaryMatrix = [  # A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
            [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],  # a
            [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],  # r
            [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],  # n
            [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],  # d
            [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],  # c
            [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],  # q
            [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],  # e
            [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],  # g
            [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],  # h
            [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],  # i
            [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],  # l
            [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],  # k
            [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],  # m
            [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],  # f
            [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],  # p
            [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],  # s
            [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],  # t
            [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],  # w
            [1.38, 1.48, 0.8, -0.56, 0.0, -0.68, -0.31, 1.03, -0.05, 0.53],  # y
            [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65], ]  # v 20*10
        binaryMatrix = autoNormTenFactor(np.array(binaryMatrix))
        WHmatrix = np.matmul(matrix2, binaryMatrix)  # L*10
        matrix_20 = matrix.sum(axis=0) / matrix.shape[0]  #
        reMatrix[i, :] =np.concatenate((matrix_20,getACC(WHmatrix,10),getDWT(matrix2),getEDT(matrix,eval(sys.argv[1]))),axis=0)
    print(reMatrix.shape)
    return reMatrix

def main():
    x1 = getMatrix("data/Train915/result/negative/pssm_profile_uniref50")
    x2 = getMatrix("data/Train915/result/positive/pssm_profile_uniref50")
    y = [-1 for i in range(x1.shape[0])]
    y.extend([1 for i in range(x2.shape[0])])
    y = np.array(y)
    x = np.vstack((x1, x2))

    test_x1 = getMatrix("data/Test850/result/negative/pssm_profile_uniref50")
    test_x2 = getMatrix("data/Test850/result/positive/pssm_profile_uniref50")

    test_x = np.vstack((test_x1, test_x2))
    test_y = [-1 for i in range(test_x1.shape[0])]
    test_y.extend([1 for i in range(test_x2.shape[0])])
    test_y=np.array(test_y)
    x_all=np.vstack((x,test_x))
    x_all=Norm(x_all)
    x = x_all[:915, :]
    test_x = x_all[915:, :]
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
    loo_probas_y = []#
    loo_test_y = []#
    loo_predict_y = []#
    for train, test in loo.split(x):
        clf.fit(x[train], y[train])
        loo_predict_y.extend(clf.predict(x[test]))  #
        loo_probas_y.extend(clf.predict_proba(x[test]))#
        loo_test_y.extend(y[test])#
    loo_probas_y = np.array(loo_probas_y)
    loo_test_y=np.array(loo_test_y)
    print(loo_probas_y.shape)
    #np.savetxt("915-RFclassification-ALL-LOO-probas_y.csv",loo_probas_y,delimiter=",")
    #np.savetxt("915-RFclassification-ALL-LOO-test_y.csv",loo_test_y,delimiter=",")
    #
    confusion = sklearn.metrics.confusion_matrix(loo_test_y, loo_predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC:{}".format(roc_auc_score(loo_test_y,loo_probas_y[:, 1])))
    print("SP:{}".format(TN / (TN + FP)))
    print("SN:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE:{}".format(TP/(TP+FP)))
    print("MCC:{}".format(n))
    print("F-score:{}".format((2*TP)/(2*TP+FP+FN)))
    print("ACC:{}".format((TP+TN)/(TP+FP+TN+FN)))


    #
    clf = gs.best_estimator_
    clf.fit(x, y)
    predict_y = clf.predict(test_x)
    print("IND：{}".format(accuracy_score(test_y, predict_y)))

main()