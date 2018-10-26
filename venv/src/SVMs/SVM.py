import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

C = 1

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex6/ex6/ex6data1.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/ex6data1.mat')

    X = data['X']
    y = data['y']

    # X = np.insert(X, 0, 1, axis=1)

    return X, y


# def sigmoid(theta, X):
#     return 1 / (1 + np.exp(-np.dot(X, theta)))
#
# def costreg(theta, X, y, C):
#     cost = -C * (np.dot(y.T, np.log(sigmoid(theta, X))) + np.dot((1-y.T), np.log(1-sigmoid(theta, X))))
#
#     return cost + 0.5 * np.sum(theta[1:]**2)
#
# def gradientreg(theta, X, y, C):
#     gra = C * np.dot((sigmoid(theta, X) - y.T), X)
#     reg = 0.5 * theta[1:]
#     reg = np.insert(reg, 0, 0, axis=0)
#
#     return gra + reg

def svmboundary(X, y, c):
    clf = svm.SVC(C=c, kernel='linear', gamma='auto')
    clf.fit(X, y.flatten())
    x1 = np.linspace(0, 4.5, 50)
    x2 = np.linspace(1.5, 5, 50)
    xx, yy = np.meshgrid(x1, x2)
    print(xx.shape, yy.shape)
    z = clf.predict([xx.flatten(), yy.flatten()])
    z = z.reshape(xx.shape)
    plot.contour(xx, yy, z)


if __name__ == '__main__':
    X, y = getDataSet()
    svmboundary(X, y, C)

    # theta = np.zeros(X.shape[1])
    # result = opt.fmin_tnc(func=costreg, x0=theta, fprime=gradientreg, args=(X, y, C))
    # opt_theta = result[0]
    # print(opt_theta)
    #
    # x1 = np.linspace(0, 4.5, 50)
    # x2 = -(opt_theta[0] + opt_theta[1]*x1) / opt_theta[2]



    # plot.plot(x1, x2, c='r')
    # for i in range(y.shape[0]):
    #     if y[i] == 0:
    #         plot.scatter(X[i, 1], X[i, 2], marker='o', c='y')
    #     else:
    #         plot.scatter(X[i, 1], X[i, 2], marker='x', c='b')
    plot.show()