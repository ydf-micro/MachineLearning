import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
from scipy.optimize import minimize

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex8/ex8/ex8data1.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex8\ex8\ex8data1.mat')

    X = data['X']
    Xval = data['Xval']
    yval = data['yval']


    return X, Xval, yval

def getHighDimensionalDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex8/ex8/ex8data2.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex8\ex8\ex8data2.mat')

    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    return X, Xval, yval

def estimateGaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)

    return mu, sigma2

def Gaussian(X, mu, sigma2):
    norm = np.power(2 * np.pi * sigma2, -0.5)
    exp = np.zeros(X.shape)
    p = np.ones((X.shape[0], 1))

    for row in range(X.shape[0]):
        exp[row] = np.exp(-0.5 * (X[row]-mu)**2 / sigma2)
    feature_p = norm * exp
    for col in range(feature_p.shape[1]):
        p[:, 0] *= feature_p[:, col]

    return p

def computeF1(cv, yval):
    '''TP(true positive)    FN(false negative)   FP(false positive)'''

    TP = len([i for i in range(len(yval)) if cv[i] and yval[i]])
    FP = len([i for i in range(len(yval)) if cv[i] and not yval[i]])
    FN = len([i for i in range(len(yval)) if not cv[i] and yval[i]])

    prec = TP / (TP + FP) if TP+FP else 0
    rec = TP / (TP + FN) if TP+FN else 0
    F1 = 2*prec*rec / (prec+rec) if prec+rec else 0

    return F1

def selectThreshold(Xval, mu, sigma2, yval):
    bestEpsilon, bestF1 = .0, .0
    pval = Gaussian(Xval, mu, sigma2)
    epsilons = np.linspace(np.min(pval), np.max(pval), 1000)

    for epsilon in epsilons:
        cv = pval < epsilon
        F1 = computeF1(cv, yval)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

def plotFigure(X, mu, sigma2, result):
    x = np.linspace(0, 30, 100)
    y = np.linspace(0, 30, 100)

    XX, YY = np.meshgrid(x, y)
    contour = np.c_[XX.flatten(), YY.flatten()]

    p = Gaussian(contour, mu, sigma2)
    p = p.reshape(XX.shape)

    plot.contour(XX, YY, p, [10**h for h in range(-20, 0, 3)], colors='k')
    for i in range(len(X)):
        if result[i]:
            plot.scatter(X[i, 0], X[i, 1], s=100, marker='o', c='', edgecolors='r')
        plot.scatter(X[i, 0], X[i, 1], marker='x', c='b')
    plot.xlabel('Latency(ms)')
    plot.ylabel('Throughput(mb/s)')
    plot.xlim((0, 30))
    plot.ylim((0, 30))
    plot.show()

def HighDimensionalDataSet():
    '''High dimensional dataset'''
    X, Xval, yval = getHighDimensionalDataSet()
    mu, sigma2 = estimateGaussian(X)
    epsilon, F1 = selectThreshold(Xval, mu, sigma2, yval)
    probability = Gaussian(X, mu, sigma2)
    result = probability < epsilon
    anomalies = len([1 for i in result if i])
    print('epsilon is {:.3}'.format(epsilon) + ', and {} amomalies found'.format(anomalies))


if __name__ == '__main__':
    X, Xval, yval = getDataSet()
    mu, sigma2 = estimateGaussian(X)
    epsilon, F1 = selectThreshold(Xval, mu, sigma2, yval)
    print(epsilon, F1)
    probability = Gaussian(X, mu, sigma2)
    result = probability < epsilon
    plotFigure(X, mu, sigma2, result)

    HighDimensionalDataSet()

    '''以下是测试部分'''
    test = np.array([[25, 25], [15, 15]])
    probability = Gaussian(test, mu, sigma2)
    print(probability < epsilon)


