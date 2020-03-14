from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier as kNN


def dataGet():
    datas = datasets.load_iris()
    data = datas.data
    label = datas.target

    feature_train, feature_test, target_train, target_test = train_test_split(data, label)
    return feature_train, feature_test, target_train, target_test


def classify(target, data, label, k):
    data_size = data.shape[0]
    diffMat = np.tile(target, (data_size, 1)) - data
    diffMat = diffMat ** 2
    sumMat = diffMat.sum(axis=1)
    result = sumMat ** 0.5
    sortedresult = result.argsort()
    classCount = {}
    for i in range(k):
        votelabel = label[sortedresult[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    feature_train, feature_test, target_train, target_test = dataGet()
    ans = classify(feature_test[0], feature_train, target_train, 5)
    print(ans)
