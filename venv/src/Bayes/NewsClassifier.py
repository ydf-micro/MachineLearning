'''
    复旦新闻语料中文文本分类
'''
import os
import re
import time
import jieba
import itertools
import numpy as np
from random import choice
import jieba.posseg

'''create dataset and class label'''
def loadDataSet():
    docList, classList = [], []
    dirlist = ['C3-Art', 'C4-Literature', 'C5-Education', 'C6-Philosophy', 'C7-History']
    for i in range(len(dirlist)):
        essaylist = sorted(os.listdir('/home/y_labor/fudanCorpus/utf/train/%s' % dirlist[i]))
        for j in range(10):
            wordList = textParse(open('/home/y_labor/fudanCorpus/utf/train/%s/%s' % (dirlist[i], essaylist[j]), encoding='UTF-8').read())
            docList.append(wordList)
            classList.append(i)

    return docList, classList

'''segmentation word by regular expression and jieba'''
def textParse(str_doc):
    regular = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\$$^_`{|}~ ]+'
    str_doc = re.sub(regular, '', str_doc)

    '''stopwords'''
    stwdlist = set([line.strip() for line in open('/home/y_labor/fudanCorpus/utf/stopwords.txt', 'r', encoding='UTF-8')])
    sent_list = str_doc.split('\n')
    partOfspeech = ['n', 'v', 'a', 'ns', 'nr', 'nt']

    word_2list = [rm_tokens([word+'/'+flag+' ' for part in sent_list for word, flag in jieba.posseg.cut(part) if flag in partOfspeech], stwdlist)]
    word_list = list(itertools.chain(*word_2list))      #merge vocabulary


    return word_list

'''remove stopwords, num, special symbol'''
def rm_tokens(words, stwdlist):
    words_list = list(words)
    for i in range(words_list.__len__()):
        word = words_list[i]
        if word in stwdlist:
            words_list.pop(i)
        elif len(word) == 1:
            words_list.pop(i)
        elif word == ' ':
            words_list.pop(i)

    return words_list

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet.union(document)

    return list(vocabSet)

'''document word bag model, and create matrix dataset'''
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = list(np.zeros(len(vocabList)))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec

'''naive bayes model training dataset and optimizating'''
def trainNBM(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    p1Num = np.ones(numWords)
    p2Num = np.ones(numWords)
    p3Num = np.ones(numWords)
    p4Num = np.ones(numWords)
    p5Num = np.ones(numWords)

    class1 = np.array([2])
    class2 = np.array([2])
    class3 = np.array([2])
    class4 = np.array([2])
    class5 = np.array([2])         #Laplace correction

    pNumList = np.array([p1Num, p2Num, p3Num, p4Num, p5Num])
    NumClass = np.array([class1, class2, class3, class4, class5])

    for i in range(numTrainDocs):
        for j in range(5):
            if trainCategory[i] == j:
                pNumList[j] += trainMatrix[i]
                NumClass[j][0] += 1

    pVect = np.log(pNumList) / NumClass
    pi = (NumClass / (numTrainDocs+10)).flatten()

    return pVect, pi

'''classifier'''
def classify(classify2vec, pVect, pi):
    result = []
    for i in range(5):
        result.append(np.sum(np.array(classify2vec) * pVect[i]) + np.log(pi[i]))
    # print(result)

    return result.index(max(result))

'''load testing dataset'''
def loadtestDataSet():
    docList, classList = [], []
    dirlist = ['C3-Art', 'C4-Literature', 'C5-Education', 'C6-Philosophy', 'C7-History']
    for i in range(len(dirlist)):
        testEssay = choice(os.listdir('/home/y_labor/fudanCorpus/utf/test/%s' % dirlist[i]))
        wordList = textParse(
            open('/home/y_labor/fudanCorpus/utf/test/%s/%s' % (dirlist[i], testEssay), encoding='UTF-8').read())
        docList.append(wordList)
        classList.append(i)

    return docList, classList

'''verification test dataset'''
def testDataSet(vocabList, pVect, pi):

    dirlist = ['C3-Art', 'C4-Literature', 'C5-Education', 'C6-Philosophy', 'C7-History']
    testList, testClass = loadtestDataSet()
    for i in range(len(testList)):
        result = classify(bagOfWord2Vec(vocabList, testList[i]), pVect, pi)
        print('计算分类结果为： {}, 实际结果为： {}'.format((dirlist[result])[3:], (dirlist[testClass[i]])[3:]))

if __name__ == '__main__':
    start = time.clock()
    docList, classList = loadDataSet()
    vocabList = createVocabList(docList)
    trainMatrix = []
    for essay in docList:
        trainMatrix.append(bagOfWord2Vec(vocabList, essay))

    pVect, pi = trainNBM(trainMatrix, classList)
    trainTime = time.clock()
    print('训练耗时：{:.3}s'.format(trainTime - start))

    testDataSet(vocabList, pVect, pi)

    end = time.clock()
    print('测试耗时：{:.3}s'.format(end-trainTime))
