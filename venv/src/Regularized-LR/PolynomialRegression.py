import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import scipy.optimize as opt

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

def costreg(theta, X, y, lamda):
    return cost(theta, X, y) + lamda*np.sum(theta[1:]**2)/(2*len(X))

def gradient(theta, X, y):
    h = np.dot(X, theta.T)
    return np.dot((h-y.flatten()).T, X) / len(X)

def gradientreg(theta, X, y, lamda):
    grad = gradient(theta, X, y)
    temp = np.copy(theta)   #这里用theta的拷贝，不改变原来的theta
    temp[0] = 0
    return grad + lamda*temp/len(X)


def train(X, y, lamda):
    theta = np.zeros(X.shape[1])
    min = opt.minimize(fun=costreg, x0=theta, jac=gradientreg, method='TNC', args=(X, y, lamda))
    return min.x

def learning_curve(X, y, Xval, yval, lamda):
    length = np.arange(1, len(X)+1)
    train_cost = []
    cross_valid_cost = []

    for i in length:
        train_theta = train(X[:i], y[:i], lamda)
        train_cost.append(costreg(train_theta, X[:i], y[:i], 0))
        cross_valid_cost.append(costreg(train_theta, Xval, yval, 0))

    return train_cost, cross_valid_cost

def learning_curve_lamda(X, y, Xval, yval, lamda):
    train_cost = []
    cross_valid_cost = []

    for i in lamda:
        train_theta = train(X, y, i)
        train_cost.append(costreg(train_theta, X, y, 0))
        cross_valid_cost.append(costreg(train_theta, Xval, yval, 0))

    print(lamda[np.argmin(cross_valid_cost)])

    return train_cost, cross_valid_cost

def polyFeatures(X):
    X_poly = np.copy(X)
    for i in range(1, 6):
        X_poly = np.hstack((X_poly, np.power(X[:, 1:], i+1)))

    return X_poly

def get_mean_std(X):
    u_X = np.mean(X, axis=0)
    s_X = np.std(X, axis=0, ddof=1)
    return u_X, s_X

def FeatureScaling(X, u, s):

    X_Scal = np.copy(X)

    X_Scal[:, 1:] = (X_Scal[:, 1:] - u[1:]) / s[1:]

    return X_Scal

if __name__ == '__main__':
    lamda = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    init_lamda = 0
    X, y, Xval, yval, Xtest, ytest = getDataSet()
    X_poly = polyFeatures(X)
    u_X, s_X = get_mean_std(X_poly)
    X_poly = FeatureScaling(X_poly, u_X, s_X)
    opt_theta = train(X_poly, y, init_lamda)

    Xval_poly = polyFeatures(Xval)
    Xval_poly = FeatureScaling(Xval_poly, u_X, s_X)
    train_cost, cross_valid_cost = learning_curve(X_poly, y, Xval_poly, yval, init_lamda)

    xx = np.arange(-80, 60, 1.4)
    xx = xx.reshape(-1, 1)
    xx = np.insert(xx, 0, 1, axis=1)
    xx_poly = polyFeatures(xx)
    xx_poly = FeatureScaling(xx_poly, u_X, s_X)

    train_cost_lamda, cross_valid_cost_lamda = learning_curve_lamda(X_poly, y, Xval_poly, yval, lamda)

    #绘图
    fig = plot.figure(num=3, figsize=(25, 20))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax1.plot(xx[:, 1:], np.dot(xx_poly, opt_theta.T), c='r')
    ax1.scatter(X[:, 1:], y)

    ax2.plot(np.arange(1, len(X)+1), train_cost, c='g', label='Train')
    ax2.plot(np.arange(1, len(X)+1), cross_valid_cost, c='b', label='Cross Validation')
    ax2.legend(['Train', 'Cross Validation'])

    ax3.plot(lamda, train_cost_lamda, c='g', label='Train')
    ax3.plot(lamda, cross_valid_cost_lamda, c='b', label='Cross Validation')
    ax3.legend(['Train', 'Cross Validation'])

    plot.show()

