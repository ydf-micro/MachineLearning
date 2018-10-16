import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from numpy import *

def getdataSet():
    path = '/home/y_labor/ml/machine-learning-ex1/ex1/ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    return data

def splitdataSet(data):
    cols = data.shape[1]
    x_values = data.iloc[:,:cols-1]
    y_values = data.iloc[:,cols-1:cols]

    x_matrix = np.matrix(x_values)
    y_matrix = np.matrix(y_values)
    return x_matrix, y_matrix

def computeCost(x_matrix, y_matrix, theta):
    hyp = (x_matrix*theta.T - y_matrix).T * (x_matrix*theta.T - y_matrix)
    return sum(hyp) / (2*len(y_matrix))

def gradientDescent(x_matrix, y_matrix, alpha, theta):
    cost = []

    temp = (alpha/len(y_matrix))*(x_matrix*theta.T-y_matrix).T*x_matrix
    while np.max(np.abs(temp)) > 0.1E-5:
        theta -= temp
        cost.append(computeCost(x_matrix, y_matrix, theta))
        temp = (alpha / len(y_matrix)) * (x_matrix * theta.T - y_matrix).T * x_matrix

    return theta, cost

if __name__ == '__main__':
    alpha = 0.05
    data = getdataSet()
    data = (data - data.mean()) / data.std()  #特征缩放
    data.insert(0, 'X_0', 1)

    theta = np.matrix(np.zeros([1, data.shape[1] - 1]))
    x_matrix, y_matrix = splitdataSet(data)

    theta, cost = gradientDescent(x_matrix, y_matrix, alpha, theta)
    print(theta)

    x1 = np.linspace(data.Size.min(), data.Size.max(), 100)
    x2 = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
    f = theta[0,0] + (theta[0, 1] * x1) + (theta[0,2] * x2)

    fig = plot.figure()
    num = arange(len(cost))
    plot.plot(num, cost, c = 'r')
    plot.show()