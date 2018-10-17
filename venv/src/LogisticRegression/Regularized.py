import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import scipy.optimize as opt

'''
    mapFeature(x) = [1, x1, x2, x1**2, x1x2, x2**2, x1**3, ..., x1x2**5, x2**6]
    cost function j(theta) = -sum(y * log(h(x)) + (1-y) * log(1-h(x)))/m + lamda*sum(theta**2)/2m 注意这里的theta从1开始，不惩罚第0个
    gradient: = sum((h(x)-y)x)/m + lamda * theta / m 同理，这里的theta也是从1开始
    
    opt.fmin_tnc函数对应matlab中的fminunc函数
'''

init_lamda = 0.1

def getdataSet():
    path = '/home/y_labor/ml/machine-learning-ex2/ex2/ex2data2.txt'
    data = pd.read_csv(path, names=['Exam1score', 'Exam2score', 'Admitted'])
    # data.insert(0, 'X_0', 1)
    return data

def feature_mapping(x1, x2, power):
    data = {'f{}{}'.format(i-p, p): np.power(x1, i-p) * np.power(x2, p)
            for i in range(0, power+1)
            for p in range(0, i+1)}
    return pd.DataFrame(data)

def splitdataSet(data):
    cols = data.shape[1]
    x_values = data.iloc[:,:cols-1].values
    y_values = data.iloc[:,cols-1:cols].values

    return x_values, y_values

def sigmoid(theta, x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

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

def predict(theta, x):
    sig = sigmoid(theta, x)
    return [1 if x >= 0.5 else 0 for x in sig]


if __name__ == '__main__':
    data = getdataSet()
    lamda = 1
    x, y = splitdataSet(data)
    x = feature_mapping(x[:, 0], x[:, 1], 6)
    theta = np.zeros(x.shape[1])


    result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x, y))
    opt_theta = result[0]

    predictions = predict(opt_theta, x)
    verify = [1 if predict == turth else 0 for (predict, turth) in zip(predictions, y)]
    currency = sum(verify) / len(y)
    print(currency)

    x = np.linspace(-0.75, 1, 200)
    x1, x2 = np.meshgrid(x, x)
    f = feature_mapping(x1.ravel(), x2.ravel(), 6)
    f = np.dot(f, opt_theta)
    f = f.reshape(x1.shape)
    plot.contour(x1, x2, f, 0, colors='r')

    plot.xlabel('Microchip Test 1')
    plot.ylabel('Microchip Test 2')
    admitted = data.groupby('Admitted')
    for category, group in admitted:
        if category == 0:
            plot.scatter(group['Exam1score'], group['Exam2score'], marker='x')
        else:
            plot.scatter(group['Exam1score'], group['Exam2score'], marker='o')
    plot.show()