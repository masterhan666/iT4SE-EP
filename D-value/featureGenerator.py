import numpy as np
from cmath import *
import pywt
from scipy.fftpack import dct
#
def get20Aminos(matrix):
    vector=matrix.sum(axis=0)/matrix.shape[0]
    return vector

#
def getACC(matrix,lag):
    ls=[]
    for i in range(matrix.shape[1]):
        a=np.sum(matrix[:,i])/matrix.shape[0]#
        for j in range(matrix.shape[1]):
            b=np.sum(matrix[:,j])/matrix.shape[0]#
            for g in range(1,lag+1):#
                n=0
                for z in range(matrix.shape[0]-g):#
                    n+=(matrix[z,i]-a)*(matrix[z+g,j]-b)
                n=n/(matrix.shape[0]-g)
                ls.append(n)
    return np.array(ls)

def getEDT(matrix,lag):
    ls=[]
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            for g in range(1,(lag+1)):
                n=0
                for z in range(matrix.shape[0]-g):
                    n+=(matrix[z,i]-matrix[z+g,j])**2
                n=n/(matrix.shape[0]-g)
                ls.append(n)
    return np.array(ls)

#
def fun(x,n,ls):
    CA,CD= pywt.dwt(x,'bior3.3')
    n-=1
    if n>=0:
        CC = dct(CA)
        ls.extend(CC[:5])
        ls.extend([min(CA),max(CA),np.mean(CA),np.std(CA),min(CD),max(CD),np.mean(CD),np.std(CD)])
        fun(CA,n,ls)
    else:
        return ls
def getDWT(matrix):
    ls=[]
    for i in range(matrix.shape[1]):
        fun(matrix[:,i],4,ls)
    return np.array(ls)















