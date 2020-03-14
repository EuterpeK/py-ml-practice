from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
    xArr = []
    yArr = []
    with open(filename) as fp:
        lines = fp.readlines()
        for line in lines:
            newline = line.strip().split('\t')
            xArr.append(list(map(float, newline[:-1])))
            yArr.append(float(newline[-1]))
    return xArr, yArr


def regularize(xMat, yMat):
    inx = xMat.copy()
    iny = yMat.copy()
    ymean = np.mean(iny, axis=0)
    inymat = iny - ymean
    xmean = np.mean(inx, axis=0)
    xvar = np.var(inx, axis=0)
    inxmat = (inx - xmean) / xvar

    return inxmat, inymat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowertError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowertError:
                    lowertError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def plotstageWiseMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadData('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotstageWiseMat()
