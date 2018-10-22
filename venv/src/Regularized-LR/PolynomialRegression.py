import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt

def getDataSet():

    #linux下
    # data = loadmat('/home/y_labor/ml/machine-learning-ex5/ex5/ex5data1.mat')

    #windows下
    data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex5\ex5\ex5data1.mat')

    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']


    # X = np.insert(X, 0, 1, axis=1)
    # Xval = np.insert(Xval, 0, 1, axis=1)
    # Xtest = np.insert(Xtest, 0, 1, axis=1)

    # print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)


    return X, y, Xval, yval, Xtest, ytest

def polyFeatures(X):
    X_poly = np.copy(X).flatten()
    for i in range(1, 8):
        temp = np.power(X.flatten(), i+1)
        X_poly = np.vstack((X_poly, temp))

    X_poly = np.insert(X_poly, 0, 1, axis=1)

    return X_poly

if __name__ == '__main__':
    X, y, Xval, yval, Xtest, ytest = getDataSet()
    print(polyFeatures(X))