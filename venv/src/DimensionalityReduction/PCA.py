import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex7/ex7/ex7data1.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex7/ex7/ex7data1.mat')

    X = data['X']

    return X

def featureNormalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    X_norm = (X - mean) / std

    return X_norm, mean, std

def pca(X):
    sigma = np.dot(X.T, X) / len(X)
    U, S, V = np.linalg.svd(sigma)

    return U, S, V

def projectData(X, U, K):
    return np.dot(X, U[:, :K])

def recoverData(Z, U, K):
    return np.dot(Z, U[:, :K].T)

if __name__ == '__main__':
    X = getDataSet()
    X_norm, mean, std = featureNormalize(X)
    U, S, V = pca(X_norm)
    # print(U)
    Z = projectData(X_norm, U, 1)
    # print(Z[0])
    X_rec = recoverData(Z, U, 1)
    # print(X_rec[0])

    fig = plot.figure(num=2, figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(X[:, 0], X[:, 1], c='', edgecolors='b', marker='.')
    ax1.plot([mean[0], 1.5 * S[0] * U[0, 0] + mean[0]], [mean[1], 1.5 * S[0] * U[0, 1] + mean[1]], c='r',
             label='frist PC')
    ax1.plot([mean[0], 1.5 * S[1] * U[1, 0] + mean[0]], [mean[1], 1.5 * S[1] * U[1, 1] + mean[1]], c='k',
             label='second PC')
    ax1.legend()

    ax2.scatter(X_norm[:, 0], X_norm[:, 1], c='', edgecolors='b', marker='.', label='original data')
    ax2.scatter(X_rec[:, 0], X_rec[:, 1], c='', edgecolors='r', marker='.', label='projected data')
    for i in range(X_norm.shape[0]):
        ax2.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], 'k--')
    ax2.legend()

    plot.axis('equal')
    plot.show()