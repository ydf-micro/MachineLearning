import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
from scipy.optimize import minimize

'''
    minimize函数中的fun的函数各个变量必须是一维的
'''

lamda_init = 1

def getDataSet():

    #linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex5/ex5/ex5data1.mat')

    #windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex5\ex5\ex5data1.mat')

    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']


    X = np.insert(X, 0, 1, axis=1)
    Xval = np.insert(Xval, 0, 1, axis=1)
    Xtest = np.insert(Xtest, 0, 1, axis=1)

    # print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)


    return X, y, Xval, yval, Xtest, ytest

def cost(theta, X, y):
    h = np.dot(X, theta.T)
    return np.sum((h-y.flatten())**2) / (2*len(X))

def costreg(theta, X, y, lamda=lamda_init):
    return cost(theta, X, y) + lamda*np.sum(theta[1:]**2)/(2*len(X))

def gradient(theta, X, y):
    h = np.dot(X, theta.T)
    return np.dot((h-y.flatten()).T, X) / len(X)

def gradientreg(theta, X, y, lamda=lamda_init):
    grad = gradient(theta, X, y)
    theta[0] = 0
    return grad + lamda*theta/len(X)

if __name__ == '__main__':
    X, y, Xval, yval, Xtest, ytest = getDataSet()
    theta = np.ones(X.shape[1])
    print(costreg(theta, X, y))
    print(gradientreg(theta, X, y))

    min = minimize(fun=costreg, x0=theta, jac=gradientreg, method='TNC', args=(X, y, lamda_init))


    plot.scatter(X[:, 1:], y)
    plot.show()