from featureGenerator import *
from readToMatrix import *
import numpy as np
import re
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, roc_curve,confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sys

#标准化物理化学性质矩阵
def autoNormTenFactor(matrix):
    for i in range(matrix.shape[1]):#lie
        mini=min(matrix[:,i])
        maxi=max(matrix[:,i])
        for j in range(matrix.shape[0]):#hang
            matrix[j][i]=(matrix[j,i]-mini)/(maxi-mini)
    return matrix
#标准化特征矩阵
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
    reMatrix = np.zeros((m, 2060+400*4))
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
        matrix_20 = matrix.sum(axis=0) / matrix.shape[0]  # 得到20维列平均
        reMatrix[i, :] =np.concatenate((matrix_20,getACC(WHmatrix,10),getDWT(matrix2),getEDT(matrix,4)),axis=0)
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
    #np.savetxt("ZIJI915x.csv",x,delimiter=",")
    #np.savetxt("ZIJI915test_x.csv",test_x,delimiter=",")
    # 降维
    clf = RandomForestClassifier(random_state=134)
    clf.fit(x,y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(indices)
    #np.savetxt("ZIJI915feaimpindices.csv",indices,delimiter=",")

    x_trans = x[:,indices[:320]]
    test_x_trans=test_x[:,indices[:320]]
    print(x_trans.shape,320)

    # 开始调优
    CC = []
    gammas = []
    for i in range(-5, 16, 2):
        CC.append(2 ** i)
    for i in range(3, -16, -2):
        gammas.append(2 ** i)
    param_grid = {"C": CC, "gamma": gammas}
    gs = GridSearchCV(SVC(probability=True), param_grid, cv=10)
    gs.fit(x_trans, y)
    print(gs.best_estimator_)
    print(gs.best_score_)

    # 开始训练
    clf = gs.best_estimator_
    loo = LeaveOneOut()
    score = cross_val_score(clf, x_trans, y, cv=loo).mean()
    print("留一法验证结果为{}".format(score))
    # 叠加probas和predict和test
    loo_probas_y = []  # 叠加阳阴预测概率
    loo_test_y = []  # 叠加label
    loo_predict_y = []  # 叠加预测值
    for train, test in loo.split(x_trans):
        clf.fit(x_trans[train], y[train])
        loo_predict_y.extend(clf.predict(x_trans[test]))  # 叠加预测结果
        loo_probas_y.extend(clf.predict_proba(x_trans[test]))  # 叠加预测值
        loo_test_y.extend(y[test])  # 叠加label
    loo_probas_y = np.array(loo_probas_y)
    loo_test_y = np.array(loo_test_y)
    np.savetxt("915-SVM-RF320-LOO-probas_y.csv", loo_probas_y, delimiter=",")
    np.savetxt("915-SVM-RF320-LOO-test_y.csv", loo_test_y, delimiter=",")
    print(loo_probas_y.shape)

    # 得到混淆矩阵，计算各种预测值
    confusion = confusion_matrix(loo_test_y, loo_predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC曲线下面积评分为:{}".format(roc_auc_score(loo_test_y, loo_probas_y[:, 1])))
    print("特异性SP为:{}".format(TN / (TN + FP)))
    print("敏感性SN为:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE为:{}".format(TP / (TP + FP)))
    print("MCC为:{}".format(n))
    print("F-score为:{}".format((2 * TP) / (2 * TP + FP + FN)))
    print("ACC为:{}".format((TP + TN) / (TP + FP + TN + FN)))


    #独立集检验
    clf = gs.best_estimator_
    clf.fit(x_trans, y)
    predict_y = clf.predict(test_x_trans)
    probas_y = clf.predict_proba(test_x_trans)
    print("独立集检验准确率为：{}".format(accuracy_score(test_y, predict_y)))

    np.savetxt("915-SVM-RF320-IND-probas_y.csv", loo_probas_y, delimiter=",")
    np.savetxt("915-SVM-RF320-IND-test_y.csv", loo_test_y, delimiter=",")

    # 得到混淆矩阵，计算各类指标
    confusion = confusion_matrix(test_y, predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC曲线下面积评分为:{}".format(roc_auc_score(test_y, probas_y[:, 1])))
    print("特异性SP为:{}".format(TN / (TN + FP)))
    print("敏感性SN为:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE为:{}".format(TP / (TP + FP)))
    print("MCC为:{}".format(n))
    print("F-score为:{}".format((2 * TP) / (2 * TP + FP + FN)))
    print("ACC为:{}".format((TP + TN) / (TP + FP + TN + FN)))

main()
'''
[3650 1651 2688 ... 2347 2345    0]
(915, 320) 320
SVC(C=128, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001953125, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
0.926775956284153
留一法验证结果为0.9245901639344263
(915, 2)
ROC曲线下面积评分为:0.9320720236495565
特异性SP为:0.9639344262295082
敏感性SN为:0.8459016393442623
PRE为:0.9214285714285714
MCC为:0.8284114115683391
F-score为:0.882051282051282
ACC为:0.9245901639344263
独立集检验准确率为：0.9564705882352941
ROC曲线下面积评分为:0.9626322580645161
特异性SP为:0.9625806451612903
敏感性SN为:0.8933333333333333
PRE为:0.6979166666666666
MCC为:0.7669910790893719
F-score为:0.783625730994152
ACC为:0.9564705882352941
'''

'''自己电脑
(610, 3660)
(305, 3660)
(775, 3660)
(75, 3660)
[3650 3648 2691 ... 1056 2699    0]
(915, 320) 320
SVC(C=128, gamma=0.001953125, probability=True)
0.9191829909221214
留一法验证结果为0.9158469945355191
(915, 2)
ROC曲线下面积评分为:0.9275839828003224
特异性SP为:0.9540983606557377
敏感性SN为:0.839344262295082
PRE为:0.9014084507042254
MCC为:0.8084562338863955
F-score为:0.8692699490662139
ACC为:0.9158469945355191
独立集检验准确率为：0.9423529411764706
ROC曲线下面积评分为:0.9672258064516129
特异性SP为:0.9483870967741935
敏感性SN为:0.88
PRE为:0.6226415094339622
MCC为:0.7111732423795029
F-score为:0.7292817679558011
ACC为:0.9423529411764706
'''