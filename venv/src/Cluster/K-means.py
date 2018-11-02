import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex7/ex7/ex7data2.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex7/ex7/ex7data2.mat')

    X = data['X']

    return X


def findClosestCentroids(X, centroids):
    idx = []

    for i in range(len(X)):
        minus = X[i] - centroids
        dist = minus[:, 0]**2 + minus[:, 1]**2
        index = np.argmin(dist)
        idx.append(index)

    return np.array(idx)

def computeCentroids(X, idx):
    centroids = []
    idx = idx.reshape(-1, 1)
    X_idx = np.hstack((X, idx))

    for i in range(len(np.unique(idx))):
        value = X[np.where(X_idx[:, 2] == i)].mean(axis=0)
        centroids.append(value)

    return np.array(centroids)

def kMeansInitCentroids(X):
    m = X.shape[0]
    idx = np.random.choice(m, 3)
    centroids = X[idx]

    return np.array(centroids).reshape(3, 2)

def compareidx(pre_idx, idx):
    for i in range(len(pre_idx)):
        if pre_idx[i] != idx[i]:
            return 1
    return 0

def Kmeans(X):
    centroids = kMeansInitCentroids(X)
    all_centroids = []
    all_centroids.append(centroids)
    idx = np.empty((X.shape[0], 1))
    pre_idx = np.ones((X.shape[0], 1))

    while compareidx(pre_idx, idx):
        pre_idx = np.copy(idx)
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx)
        all_centroids.append(centroids)

    # print(np.array(all_centroids))
    return idx, np.array(all_centroids)

if __name__ == '__main__':
    X = getDataSet()
    idx, all_centroids = Kmeans(X)

    plot.plot(all_centroids[:, 0, 0], all_centroids[:, 0, 1], 'x--', c='k')
    plot.plot(all_centroids[:, 1, 0], all_centroids[:, 1, 1], 'x--', c='k')
    plot.plot(all_centroids[:, 2, 0], all_centroids[:, 2, 1], 'x--', c='k')
    for i in range(len(idx)):
        if idx[i] == 0:
            plot.scatter(X[i, 0], X[i, 1], marker='.', c='r')
        elif idx[i] == 1:
            plot.scatter(X[i, 0], X[i, 1], marker='.', c='g')
        else:
            plot.scatter(X[i, 0], X[i, 1], marker='.', c='b')

    plot.show()