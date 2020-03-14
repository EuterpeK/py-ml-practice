from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


def loadData(filename):
    xArr = []
    yArr = []
    with open(filename) as fp:
        lines = fp.readlines()
        # numFeat = np.shape(lines)[0]
        for line in lines:
            newline = line.strip().split('\t')
            xArr.append(list(map(float, newline[:-1])))
            yArr.append(float(newline[-1]))
    return xArr, yArr


def ridgeRegression(xMat, yMat, lam=0.2):
    xMat = np.mat(xMat)
    yMat = np.mat(yMat)
    xTx = xMat.T * xMat
    denom = xTx + lam * np.eye(np.shape(xMat)[1])
    if np.linalg.det(denom) == 0.0:
        print('矩阵是奇异矩阵，无法求逆')
        return
    ws = denom.I * xMat.T * yMat
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegression(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def plotwMat():
    """
    函数说明:绘制岭回归系数矩阵
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-20
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadData('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotwMat()
