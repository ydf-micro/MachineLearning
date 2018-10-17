import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plot

'''
    cost function: J(theta) = -sum(sum(y * log(h(x)) + (1-y) * log(1-h(x)))/m ) + 
'''

init_lamda = 0.1

def getDataSet():
    #linux下
    # data = loadmat('/home/y_labor/ml/machine-learning-ex4/ex4/ex4data1.mat')
    # weight = loadmat('/home/y_labor/ml/machine-learning-ex4/ex4/ex4weights.mat')

    #windows下
    data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex4\ex4\ex4data1.mat')
    weight = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex4\ex4\ex4weights.mat')

    x      = data['X']
    y      = data['y']
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']


    return x, y, theta1, theta2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feedforward(x, theta1, theta2):
    hidden1 = np.dot(x, theta1.T)
    hidden1 = np.insert(hidden1, 0, 1, axis=1)
    hidden1 = sigmoid(hidden1)

    hidden2 = np.dot(hidden1, theta2.T)
    hidden2 = sigmoid(hidden2)
    print(hidden2)

def visualData(x):
    select_some = np.random.choice(np.arange(x.shape[0]), 100)
    image = x[select_some, :]
    fig, ax_array = plot.subplots(10, 10, sharex=True, sharey=True, figsize=(8, 8))
    for row in range(10):
        for col in range(10):
            ax_array[row, col].matshow(image[10*row+col].reshape(20, 20))
    plot.xticks([])
    plot.yticks([])
    plot.show()

def coding_y(y):
    coding = np.empty((y.shape[0], 10))
    print(coding.shape)
    i = 0
    for j in y:
        coding[i] = np.zeros(10)
        if j == 10:
            coding[i][j-10] = 1
        else:
            coding[i][j] = 1
        i += 1

    return coding

if __name__ == '__main__':
    x, y, theta1, theta2 = getDataSet()
    # visualData(x)
    x = np.insert(x, 0, 1, axis=1)
    # print(x.shape, theta1.shape, theta2.shape, y.shape)
    # print(coding_y(y))
    feedforward(x, theta1, theta2)