# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    data = []
    label = []
    with open('testSet.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            data.append([1.0, float(line[0]), float(line[1])])
            label.append(int(line[2]))
    return data, label


def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))


def gradAscent(data, label):
    data_matrix = np.mat(data)
    label_matrix = np.mat(label).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights.getA()


def stocGradAscent1(data, label, numlter=150):
    m, n = np.shape(data)
    weights = np.ones(n)
    for j in range(numlter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(data[randIndex] * weights))


def plotBestFit(weights):
    dataMat, labelMat = load_data()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = []
    ycord1 = []  # 正样本
    xcord2 = []
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1')
    plt.ylabel('X2')  # 绘制label
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = load_data()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
