import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

def getdataSet():
    path = '/home/y_labor/ml/machine-learning-ex1/ex1/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    data.insert(0, 'X_0', 1)
    return data

def splitdataSet(data):
    cols = data.shape[1]
    x_values = data.iloc[:,:cols-1]
    y_values = data.iloc[:,cols-1:cols]

    x_matrix = np.matrix(x_values)
    y_matrix = np.matrix(y_values)
    return x_matrix, y_matrix

'''
    cost function = sum((h(x)-y)**2)/2m 
    h(x) = x * theta(i).T  (x0 = 1)
'''
def computeCost(x_matrix, y_matrix, theta):
    hyp = np.power((x_matrix*theta.T - y_matrix), 2)
    return sum(hyp) / (2*len(y_matrix))
'''
    theta(j) = theta(j) - alpha * sum((h(x) - y) * x) / m
'''
def gradientDescent(x_matrix, y_matrix, alpha, theta):
    cost = []

    temp = (alpha/len(y_matrix))*(x_matrix*theta.T-y_matrix).T*x_matrix
    while np.max(np.abs(temp)) > 0.1E-5:
        theta -= temp
        cost.extend(computeCost(x_matrix, y_matrix, theta).tolist())
        temp = (alpha / len(y_matrix)) * (x_matrix * theta.T - y_matrix).T * x_matrix

    return theta, cost

if __name__ == '__main__':
    alpha = 0.01
    data = getdataSet()
    theta = np.matrix(np.zeros([1, data.shape[1] - 1]))
    x_matrix, y_matrix = splitdataSet(data)
    theta, cost = gradientDescent(x_matrix, y_matrix, alpha, theta)
    print(theta)
    print(cost)

    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0,0] + (theta[0, 1] * x)

    # fig = plot.figure(num=1)
    plot.plot(x, f, c = 'r')
    plot.scatter(data['Population'], data['Profit'])
    plot.show()