import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

init_lamda = 0.1

def getDataSet():
    data = loadmat('/home/y_labor/ml/machine-learning-ex3/ex3/ex3data1.mat')
    x = data['X']
    y = data['y']

    return x, y

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

def sigmoid(theta, x):
    return 1/(1 + np.exp(-np.dot(x, theta)))

def cost(theta, x, y):
    sig = sigmoid(theta, x)
    j1 = np.dot(y.T, np.log(sig))
    j2 = np.dot((1 - y.T), np.log(1 - sig))
    loss = -(j1 + j2) / len(x)
    return loss

def costReg(theta, x, y, lamda=init_lamda):
    reg = lamda * np.sum(theta[1:]**2)/(2*len(x))
    return cost(theta, x, y) + reg

def gradient(theta, x, y):
    sig = sigmoid(theta, x)
    return np.dot((sig - y.T), x) / len(x)

def gradientReg(theta, x, y, lamda=init_lamda):
    reg = (lamda/len(x)) * theta[1:]
    reg = np.insert(reg, 0, 0, axis=0)
    return gradient(theta, x, y) + reg

def one_vs_all(x, y, num, lamda=init_lamda):
    all_theta = np.zeros((num, x.shape[1]))
    for i in range(1, num+1):
        theta = np.zeros(x.shape[1])
        y_i = [1 if l == i else 0 for l in y]
        y_i = np.array(y_i)

        min = minimize(fun=costReg, x0=theta, jac=gradientReg, method='TNC', args=(x, y_i))

        all_theta[i-1, :] = min.x

    return all_theta

def predict(theta, x):
    sig = sigmoid(theta, x)

    max_sig = np.argmax(sig, axis=1)
    max_sig = max_sig.reshape(5000, 1)
    max_sig += 1

    return max_sig

if __name__ == '__main__':
    x, y = getDataSet()
    theta = one_vs_all(x, y, len(np.unique(y)))
    result = predict(theta.T, x)
    accuracy = np.mean(result == y)
    print(accuracy)
