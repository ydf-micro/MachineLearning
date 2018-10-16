from DataSet import data_set
from math import log
import treePlotter


#计算信息熵
def cal_entro(dataSet):
    num = len(dataSet)
    labels = {}

    #计算同一属性不同类别的个数
    for row in dataSet:
        label = row[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1

    Ent = 0.0
    #计算信息熵
    for key in labels:
        prob = float(labels[key]) / num
        Ent -= prob * log(prob, 2)

    return Ent

#返回第i个特征的值为value的全部数据集
def split_dataSet(dataSet, i, value):
    ret_dataSet = []
    for row in dataSet:
        if row[i] == value:
            re_row = row[:i]
            re_row.extend(row[i+1:])
            ret_dataSet.append(re_row)

    return ret_dataSet

#选择最佳的特征
def choose_best_feature(dataSet):
    baseEnt = cal_entro(dataSet)
    bestFeature = -1
    bestGain = 0.0

    num = len(dataSet[0]) - 1
    for i in range(num):
        eveEnt = 0.0
        # 按历遍历数据集，选取一个特征的全部取值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        for value in uniqueVals:
            sub_dataSet = split_dataSet(dataSet, i, value)
            prob = len(sub_dataSet) / float(len(dataSet))
            eveEnt += prob * cal_entro(sub_dataSet)

        #信息增益
        infoEnt = baseEnt - eveEnt
        if infoEnt > bestGain:
            bestFeature = i
            bestGain = infoEnt

    return bestFeature

#
def majorityCnt(classList):
    count = {}
    for value in classList:
        if value not in count.keys():
            count[value] = 0
        count[value] += 1

    sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    return count[0][0]

#建立决策树
def create_tree(dataSet, labels):
    copylabels = labels[:]
    classList = [example[-1] for example in dataSet]
    #如果只有一种类别，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #没有特征了
    if dataSet[0] == 1:
        return majorityCnt(classList)

    bestFeature = choose_best_feature(dataSet)
    bestLabel = copylabels[bestFeature] #最佳特征
    tree = {bestLabel:{}}
    del copylabels[bestFeature]

    featList = [example[bestFeature] for example in dataSet]
    uniquelVals = set(featList)

    for value in uniquelVals:
        copy2labels = copylabels[:]
        tree[bestLabel][value] = create_tree(split_dataSet(dataSet, bestFeature, value), copy2labels)

    return tree

#使用决策树来测试用例进行预测
def classify(tree, labels, testData):
    key = list(tree.keys())[0]
    dict = tree[key]
    featIndex = labels.index(key)
    for key in dict:
        if key == testData[featIndex]:
            if type(dict[key]).__name__ == 'dict':
                classLabel = classify(dict[key], labels, testData)
            else:
                classLabel = dict[key]

    return classLabel

if __name__ == '__main__':
    dataSet, labels = data_set()
    myTree = create_tree(dataSet, labels)
    print(myTree)
    treePlotter.createPlot(myTree)

    testData = ['浅白', '蜷缩', '清脆', '清晰', '平坦', '硬滑']
    result = classify(myTree, labels, testData)
    print(result)