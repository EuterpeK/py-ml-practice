import numpy as np
import random


def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))


def randAscend(data_matrix, class_matrix, numlter=150):
    m, n = np.shape(data_matrix)
    # data_matrix = np.asarray(data_matrix)
    weights = np.ones(n)
    for j in range(numlter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(data_matrix[randIndex] * weights))
            error = class_matrix[randIndex] - h
            weights = weights + alpha * error * data_matrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classify():
    train = open('horseColicTraining.txt').readlines()
    test = open('horseColicTest.txt').readlines()
    trainingSet = []
    trainingLabels = []
    for line in train:
        curline = line.strip().split('\t')
        newline = []
        for i in range(len(curline) - 1):
            newline.append(float(curline[i]))
        trainingSet.append(newline)
        trainingLabels.append(float(curline[-1]))
    trainWeights = randAscend(np.array(trainingSet), np.array(trainingLabels), 500)

    error = 0
    numTest = 0
    for line in test:
        numTest += 1
        curLine = line.strip().split('\t')
        dataarr = []
        for i in range(len(curline) - 1):
            dataarr.append(float(curLine[i]))
        if int(classifyvector(np.array(dataarr), trainWeights)) != int(curLine[-1]):
            error += 1

    errorRate = error / numTest
    print('错误率为：', errorRate)


def classifyvector(ina, weights):
    prob = sigmoid(sum(ina * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.5


if __name__ == '__main__':
    classify()
