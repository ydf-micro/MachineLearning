import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt

'''
    minimize函数中的fun的函数各个变量必须是一维的
'''

lamda_init = 0

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
    temp = np.copy(theta)   #这里用theta的拷贝，不改变原来的theta
    temp[0] = 0
    return grad + lamda*temp/len(X)

def train(X, y, lamda=lamda_init):
    theta = np.zeros(X.shape[1])
    min = opt.minimize(fun=costreg, x0=theta, jac=gradientreg, method='TNC', args=(X, y, lamda))
    return min.x

def learning_curve(X, y, Xval, yval, lamda=lamda_init):
    length = np.arange(1, len(X)+1)
    train_cost = []
    cross_valid_cost = []

    for i in length:
        train_theta = train(X[:i], y[:i], lamda)
        train_cost.append(costreg(train_theta, X[:i], y[:i], lamda))
        cross_valid_cost.append(costreg(train_theta, Xval, yval, lamda))

    return length, train_cost, cross_valid_cost

if __name__ == '__main__':
    X, y, Xval, yval, Xtest, ytest = getDataSet()
    # print(costreg(theta, X, y))
    # print(gradientreg(theta, X, y))

    opt_theta = train(X, y)
    x, train_cost, cross_valid_cost = learning_curve(X, y, Xval, yval)

    fig = plot.figure(num=2, figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(X[:, 1], np.dot(X, opt_theta.T), c='r')
    ax1.scatter(X[:, 1:], y)

    ax2.plot(x, train_cost, c='g', label='Train')
    ax2.plot(x, cross_valid_cost, c='b', label='Cross Validation')
    ax2.legend(['Train', 'Cross Validation'])

    plot.show()