import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex6/ex6/ex6data3.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/ex6data3.mat')

    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']

    return X, y, Xval, yval

def svmboundary(X, y, c, sigma):
    gam = np.power(sigma, -2, dtype=float) / 2
    clf = svm.SVC(C=c, kernel='rbf', gamma=gam)
    clf.fit(X, y.flatten())
    x1 = np.linspace(-0.5, 0.3, 500)
    x2 = np.linspace(-0.6, 0.6, 500)
    xx, yy = np.meshgrid(x1, x2)
    xy = np.c_[xx.flatten(), yy.flatten()]
    z = clf.predict(xy)
    z = z.reshape(xx.shape)
    plot.contour(x1, x2, z)

def svmPredict(X, y, Xval, yval):
    cvalues = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
    sigmavalues = cvalues
    best_score = 0
    c, sigma = 0, 0

    for C in cvalues:
        for sig in sigmavalues:
            gam = np.power(sig, -2, dtype=float) / 2
            clf = svm.SVC(C=C, kernel='rbf', gamma=gam)
            clf.fit(X, y.flatten())
            score = clf.score(Xval, yval)
            if score >= best_score:
                best_score = score
                c, sigma = C, sig

    return c, sigma

if __name__ == '__main__':
    X, y, Xval, yval = getDataSet()
    c, sigma = svmPredict(X, y, Xval, yval)
    svmboundary(X, y, c, sigma)

    for i in range(y.shape[0]):
        if y[i] == 0:
            plot.scatter(X[i, 0], X[i, 1], marker='o', c='y')
        else:
            plot.scatter(X[i, 0], X[i, 1], marker='+', c='black')
    plot.show()