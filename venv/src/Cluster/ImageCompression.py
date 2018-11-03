from skimage import io
import numpy as np
import matplotlib.pyplot as plot
import Kmeans

def getDataSet():
    # linux下
    image = io.imread('/home/y_labor/ml/machine-learning-ex7/ex7/bird_small.png')

    # windows下
    # image = io.imread('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex7/ex7/bird_small.png')

    return image/255

if __name__ == '__main__':
    image = getDataSet()
    compress_image = np.zeros(image.reshape(-1, 3).shape)
    idx, all_centroids = Kmeans.executeKmeans(image.reshape(-1, 3), 16)
    centroids = all_centroids[-1]

    for i in range(len(centroids)):
        compress_image[idx == i] = centroids[i]
    compress_image = compress_image.reshape((128, 128, 3))

    fig = plot.figure(num=2, figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image)
    ax2.imshow(compress_image)
    plot.show()