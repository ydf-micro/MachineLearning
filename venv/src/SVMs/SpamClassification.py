import re
import nltk
import numpy as np
import pandas as pd
from sklearn import svm
import scipy.optimize as opt
from scipy.io import loadmat
import matplotlib.pyplot as plot
from nltk.stem.porter import PorterStemmer

'''
    Lower-casing:
    Stripping HTML:
    Normalizing URLs:
    Normalizing Email Addresses:
    Normalizing Numbers:
    Normalizing Dollars:
    Word Stemming:
    Removal of non-words:
'''

def getDataSet():
    # linux下
    path = '/home/y_labor/ml/machine-learning-ex6/ex6/vocab.txt'
    voc_list = pd.read_csv(path, sep='\t', header=None, names=['num', 'words'])
    voc_list = dict(zip(voc_list['words'], voc_list['num']))
    spamTest = loadmat('/home/y_labor/ml/machine-learning-ex6/ex6/spamTest.mat')
    spamTrain = loadmat('/home/y_labor/ml/machine-learning-ex6/ex6/spamTrain.mat')
    # print(spamTest.keys(), spamTrain.keys())

    # windows下
    # path = 'C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/vocab.txt'
    # voc_list = pd.read_csv(path, header=None, names=['num', 'words'])
    # spamTest = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/spamTest.mat')
    # spamTrain = loadmat('C:\\Users\ydf_m\Desktop\machinelearning\machine-learning-ex6/ex6/spamTrain.mat')

    Xtest = spamTest['Xtest']
    ytest = spamTest['ytest']
    X = spamTrain['X']
    y = spamTrain['y']


    return voc_list, Xtest, ytest, X, y

def getexample():
    example = []
    with open('/home/y_labor/ml/machine-learning-ex6/ex6/emailSample1.txt') as f:
        example.append(f.read())
    with open('/home/y_labor/ml/machine-learning-ex6/ex6/emailSample2.txt') as f:
        example.append(f.read())
    with open('/home/y_labor/ml/machine-learning-ex6/ex6/spamSample1.txt') as f:
        example.append(f.read())
    with open('/home/y_labor/ml/machine-learning-ex6/ex6/spamSample2.txt') as f:
        example.append(f.read())

    return example

def processEmail(email):
    email = email.lower()
    email = re.sub(r'(<.*)?>', '', email)
    email = re.sub(r'(https?://)?www.*?[/|\s]', 'httpaddr', email)
    email = re.sub(r'[\w\d]+([._-][\w\d]+)@.+.(com|org|net)', 'emailaddr', email)
    email = re.sub(r'[\d]+', 'number', email)
    email = re.sub(r'[$]+', 'dollar', email)
    email = re.sub(r'[@$/#.-:&*+=[\]?!(){\},\'">_<;%]+', ' ', email)
    email = re.sub(r'[\t\n\s]+', ' ', email)
    email = nltk.word_tokenize(email)
    porter = PorterStemmer()
    email = [porter.stem(w) for w in email]

    return email

def word_indices(email, voc_list):
    indices = []
    for word in email:
        if word in voc_list:
            indices.append(voc_list[word])

    return indices

def emailFeatures(voc_list, indices):
    feature = np.zeros(len(voc_list))
    for i in indices:
        feature[i] = 1
    print('feature vector had length {} and {} non-zero entries'.format(len(feature), sum(feature)))

    return feature

def trainsvm(X, y, Xtest, ytest, c):
    clf = svm.SVC(C=c, kernel='linear', gamma='auto')
    clf.fit(X, y.flatten())

    predTrain = clf.score(X, y)
    predTest = clf.score(Xtest, ytest)

    print('the classifier gets a training accuracy of about {:.2%} and a test accuracy of about {:.2%}'.format(predTrain, predTest))

    return clf

def predict(example, clf):
    for email in example:
        email = processEmail(email)
        indices = word_indices(email, voc_list)
        feature = emailFeatures(voc_list, indices)
        feature = feature.reshape(1, -1)
        result = clf.predict(feature)
        if result == 0:
            print('non-spam')
        else:
            print('is spam')

if __name__ == '__main__':
    voc_list, Xtest, ytest, X, y = getDataSet()
    clf = trainsvm(X, y, Xtest, ytest, 0.5)

    example = getexample()
    predict(example, clf)