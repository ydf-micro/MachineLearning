import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import scipy.optimize as opt

def getdataSet():
    path = '/home/y_labor/ml/machine-learning-ex2/ex2/ex2data1.txt'
    data = pd.read_csv(path, names=['Exam1score', 'Exam2score', 'Admitted'])
    data.insert(0, 'X_0', 1)
    return data

def splitdataSet(data):
    cols = data.shape[1]
    x_values = data.iloc[:,:cols-1].values
    y_values = data.iloc[:,cols-1:cols].values

    return x_values, y_values

def sigmoid(theta, x):
    return 1/(1 + np.exp(-np.dot(x, theta)))

def cost(theta, x, y):
   sig = sigmoid(theta, x)
   j1 = np.dot(y.T, np.log(sig))
   j2 = np.dot((1-y.T), np.log(1-sig))
   loss = -(j1 + j2)/len(x)
   return loss


def gradient(theta, x, y):
    sig = sigmoid(theta, x)
    return np.dot((sig - y.T), x) / len(x)

def predict(theta, x):
    sig = sigmoid(theta, x)
    return [1 if x >= 0.5 else 0 for x in sig]

if __name__ == '__main__':
    data = getdataSet()
    x, y = splitdataSet(data)
    theta = np.zeros(x.shape[1])


    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    opt_theta = result[0]

    predictions = predict(opt_theta, x)
    verify = [1 if predict == turth else 0 for (predict, turth) in zip(predictions, y)]
    currency = sum(verify)/len(y)
    print(currency)


    x1 = np.arange(0, 120, 10)
    x2 = -(opt_theta[0] + opt_theta[1]*x1)/opt_theta[2]
    plot.xlabel('Exam1Score')
    plot.ylabel('Exam2Score')
    plot.xlim(20, 110)
    plot.ylim(20, 110)
    plot.plot(x1, x2, c='r')
    admitted = data.groupby('Admitted')
    for category, group in admitted:
        if category == 0:
            plot.scatter(group['Exam1score'], group['Exam2score'], marker='x')
        else:
            plot.scatter(group['Exam1score'], group['Exam2score'], marker='o')
    plot.show()