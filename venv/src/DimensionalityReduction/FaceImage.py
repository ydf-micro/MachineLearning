import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
import PCA

def getDataSet():
    # linux下
    data = loadmat('/home/y_labor/ml/machine-learning-ex7/ex7/ex7faces.mat')

    # windows下
    # data = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex7/ex7/ex7faces.mat')

    X = data['X']

    return X

def displayData(X, rows, cols):
    fig, axs = plot.subplots(rows, cols, figsize=(8, 8))
    for row in range(rows):
        for col in range(cols):
            axs[row][col].imshow(X[row * rows + col].reshape(32, 32).T, cmap=plot.cm.gray)
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

    plot.show()

if __name__ == '__main__':
    X = getDataSet()
    displayData(X, 10, 10)
    print(X.shape)
    X_norm, mean, std = PCA.featureNormalize(X)
    U, S, V = PCA.pca(X_norm)
    print(U.shape, S.shape)
    displayData(U[:, :36].T, 6, 6)
    Z = PCA.projectData(X_norm, U, 36)
    # print(Z[0])
    X_rec = PCA.recoverData(Z, U, 36)
    # print(X_rec[0])
    displayData(X_rec, 10, 10)
