import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plot
import scipy.optimize as opt
from sklearn.metrics import classification_report

'''
    Training a neural network
    1.Randomly initialize weights
    2.Implement forward propagation to get h(x(i)) for any x(i)
    3.Implement code to compute cost function J(theta)
    4.Implement backprop to compute partial derivatives
        for i = 1:m:
            Perform forward propagation and backpropagation using example(x(i), y(i))
            (get activations a(l) and delta terms for l = 2.....L)
            delta(l) = delta(l) + a(l)*delta(l+1)
    5.Use gradient checking to compare partial derivatives computed using backpropagation vs. 
      using numerical estimate of gradient of J(theta)
      Then disable gradient checking code
    6.Use gradient descent or advanced optimization method with backpropagation to try to minimize J(theta)
      as a function of parameters theta 
'''

init_lamda = 1
epsilon_init = 0.12
epsilon = 0.1E-3

def get_dataset():
    #linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex4/ex4/ex4data1.mat')
    weight = loadmat('/home/y_labor/ml/machine-learning-ex4/ex4/ex4weights.mat')

    #windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex4\ex4\ex4data1.mat')
    # weight = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex4\ex4\ex4weights.mat')

    x      = data['X']
    y      = data['y']
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']


    return x, y, theta1, theta2

def visual_data(x):
    select_some = np.random.choice(np.arange(x.shape[0]), 100)
    image = x[select_some, :]
    fig, ax_array = plot.subplots(10, 10, sharex=True, sharey=True, figsize=(8, 8))
    for row in range(10):
        for col in range(10):
            ax_array[row, col].matshow(image[10*row+col].reshape(20, 20))
    plot.xticks([])
    plot.yticks([])
    plot.show()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feedforward(x, theta):
    theta1, theta2 = roll(theta)
    hidden1_in = np.dot(x, theta1.T)
    hidden1_out = np.insert(sigmoid(hidden1_in), 0, 1, axis=1)

    output_in = np.dot(hidden1_out, theta2.T)
    output_out = sigmoid(output_in)

    return x, hidden1_in, hidden1_out, output_in, output_out

def coding_y(y):
    coding = np.empty((y.shape[0], 10))
    i = 0
    for j in y:
        coding[i] = np.zeros(10)
        coding[i][j-1] = 1
        i += 1
    return coding

def unroll(theta1, theta2):
    return np.hstack((theta1.flatten(), theta2.flatten()))

def roll(theta):
    return theta[:25*401].reshape(25, 401), theta[25*401:].reshape(10, 26)

def cost(theta, x, y):
    x, hidden1_in, hidden1_out, output_in, h = feedforward(x, theta)
    J = 0
    for i in range(len(x)):
        j1 = np.dot(y[i], np.log(h[i]))
        j2 = np.dot((1 - y[i]), np.log(1 - h[i]))
        J += -(j1 + j2)
    return J / len(x)

def regularized_cost(theta, x, y, lamda=init_lamda):
    Theta = 0
    theta1, theta2 = roll(theta)
    for i in range(theta1.shape[0]):
        Theta += sum(theta1[i, 1:] ** 2)
    for i in range(theta2.shape[0]):
        Theta += sum(theta2[i, 1:] ** 2)
    return cost(theta, x, y) + lamda * Theta / (2 * len(x))

def random_initial(size, epsilon = epsilon_init):
    return np.random.uniform(-epsilon, epsilon, size)

def partial_g(x):
    return sigmoid(x) * (1 - sigmoid(x))

def gradient(theta, x, y):
    x, hidden1_in, hidden1_out, output_in, output_out = feedforward(x, theta)
    theta1, theta2 = roll(theta)
    # print(hidden1_in.shape, hidden1_out.shape, output_in.shape, output_out.shape, theta1.shape, theta2.shape)
    delta3 = output_out - y
    delta2 = np.dot(delta3, theta2[:, 1:]) * partial_g(hidden1_in)
    partial2 = np.dot(delta3.T, hidden1_out)
    partial1 = np.dot(delta2.T, x)

    partial = unroll(partial1, partial2) / len(x)
    return partial

def regularized_gradient(theta, x, y, lamda = init_lamda):
    x, hidden1_in, hidden1_out, output_in, output_out = feedforward(x, theta)
    partial1, partial2 = roll(gradient(theta, x, y))
    theta1, theta2 = roll(theta)
    theta1[:, 0] = 0
    theta2[:, 0] = 0
    partial1 += lamda * theta1 / len(x)
    partial2 += lamda * theta2 / len(x)

    partial = unroll(partial1, partial2)

    return partial


def gradient_checking(theta, x, y, e = epsilon):
    def a_numeric_grad(plus, minus):
        return (regularized_cost(plus, x, y) - regularized_cost(minus, x, y)) / (2* e)
    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy
        minus = theta.copy
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(theta, x, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(diff)

def training(x, y):
    pass
    init_theta = random_initial(10285)

    result = opt.minimize(fun=regularized_cost, x0=init_theta, args=(x, y), method='TNC',
                          jac=regularized_gradient)
    return result

def accuracy(theta, x, y):
    x, hidden1_in, hidden1_out, output_in, output_out = feedforward(x, theta)
    y_i = np.argmax(output_out, axis=1) + 1
    print(classification_report(y, coding_y(y_i)))

if __name__ == '__main__':
    x, y, theta1, theta2 = get_dataset()
    theta = unroll(theta1, theta2)
    x = np.insert(x, 0, 1, axis=1)
    y = coding_y(y)

    result = training(x, y)
    print(result)
    accuracy(theta, x, y)
