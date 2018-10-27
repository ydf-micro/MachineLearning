import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

C = 1

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex6/ex6/ex6data2.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/ex6data2.mat')

    X = data['X']
    y = data['y']

    return X, y

def gaussianKernel(x1, x2, sigma):
    return np.exp(-np.sum((x1-x2)**2) / (2 * sigma**2))

def svmboundary(X, y, c):
    sigma = 0.1
    gam = np.power(sigma, -2, dtype=float) / 2
    clf = svm.SVC(C=c, kernel='rbf', gamma=gam)
    clf.fit(X, y.flatten())
    x1 = np.linspace(0, 1, 500)
    x2 = np.linspace(0.4, 0.9, 500)
    xx, yy = np.meshgrid(x1, x2)
    xy = np.c_[xx.flatten(), yy.flatten()]
    z = clf.predict(xy)
    z = z.reshape(xx.shape)
    plot.contour(x1, x2, z)

if __name__ == '__main__':
    X, y = getDataSet()
    svmboundary(X, y, C)

    for i in range(y.shape[0]):
        if y[i] == 0:
            plot.scatter(X[i, 0], X[i, 1], marker='o', c='y')
        else:
            plot.scatter(X[i, 0], X[i, 1], marker='+', c='black')
    plot.show()