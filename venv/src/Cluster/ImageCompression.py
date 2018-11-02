from skimage import io
import numpy as np
import matplotlib.pyplot as plot
# import K-means

def getDataSet():
    # linux下
    # image = loadmat('/home/y_labor/ml/machine-learning-ex7/ex7/bird_small.png')

    # windows下
    image = io.imread('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex7/ex7/bird_small.png')

    print(image.shape)
    plot.imshow(image)
    plot.show()

if __name__ == '__main__':
    getDataSet()