import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from scipy.io import loadmat

'''
    a = g(z)
    z = sum(x * theta.T)
'''

init_lamda = 0.1

def getDataSet():
    #linux下
    # data = loadmat('/home/y_labor/ml/machine-learning-ex3/ex3/ex3data1.mat')
    # weight = loadmat('/home/y_labor/ml/machine-learning-ex3/ex3/ex3weights.mat')

    #windows下
    data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex3\ex3\ex3data1.mat')
    weight = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex3\ex3\ex3weights.mat')

    x      = data['X']
    y      = data['y']
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']

    x      = np.insert(x, 0, 1, axis=1)

    return x, y, theta1, theta2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == '__main__':
    x, y, theta1, theta2 = getDataSet()
    # print(x.shape, theta1.shape, theta2.shape, y.shape)

    hidden1 = np.dot(x, theta1.T)
    hidden1 = np.insert(hidden1, 0, 1, axis=1)
    hidden1 = sigmoid(hidden1)

    hidden2 = np.dot(hidden1, theta2.T)
    hidden2 = sigmoid(hidden2)

    result = np.argmax(hidden2, axis=1)
    result = result.reshape(5000, 1)
    result += 1
    accuracy = np.mean(result == y)
    print(accuracy)