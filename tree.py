from math import log
import pickle


def createData():
    data = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return data, labels


def shannonent(data):
    line_num = len(data)
    labelCounts = {}
    for vec in data:
        feature = vec[-1]
        labelCounts[feature] = labelCounts.get(feature, 0) + 1
    shannon = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / line_num
        shannon -= prob * log(prob, 2)
    return shannon


def data_split(data, axis, value):
    ret = []
    for vec in data:
        if vec[axis] == value:
            reduced = vec[:axis]
            reduced.extend(vec[axis + 1:])
            ret.append(reduced)
    return ret


def choose(data):
    feature_num = len(data[0]) - 1
    base_shanon = shannonent(data)
    best_infogain = 0.0
    best_feature = -1

    for i in range(feature_num):
        featlist = [example[i] for example in data]
        unique = set(featlist)
        newEntroy = 0.0
        for value in unique:
            subdata = data_split(data, i, value)
            prob = len(subdata) / float(len(data))
            newEntroy += prob * shannonent(subdata)
        infogain = base_shanon - newEntroy

        if infogain > best_infogain:
            best_infogain = infogain
            best_feature = i

    return best_feature


def majority(classlist):
    classCount = {}
    for vote in classCount:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


def creatTree(data, labels, featLabels):
    classList = [x[-1] for x in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majority(classList)
    bestFeat = choose(data)
    bestLabel = labels[bestFeat]
    featLabels.append(bestLabel)
    myTree = {bestLabel: {}}
    del (labels[bestFeat])
    featValue = [x[bestFeat] for x in data]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        myTree[bestLabel][value] = creatTree(data_split(data, bestFeat, value), labels, featLabels)

    return myTree


def classify(tree, featLabels, vec):
    firstStr = next(iter(tree))
    secondeDict = tree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondeDict.keys():
        if vec[featIndex] == key:
            if type(secondeDict[key]).__name__ == 'dict':
                classLabel = classify(secondeDict[key], featLabels, vec)
            else:
                classLabel = secondeDict[key]
    return classLabel


def storeTree(tree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(tree, fw)


def loadTree(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


if __name__ == '__main__':
    dataSet, labels = createData()
    label = labels.copy()
    featLabels = []
    myTree = creatTree(dataSet, labels, featLabels)
    testVec = [0, 0, 1, 0]  # 测试数据
    result = classify(myTree, label, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')

    storeTree(myTree, 'class.txt')
    tree = loadTree('class.txt')
    print(tree)
