import numpy as np
from functools import reduce


def loadData():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def setwords(vocab, inputSet):
    retvec = [0] * len(vocab)
    for word in inputSet:
        if word in vocab:
            retvec[vocab.index(word)] = 1
        else:
            print('the word:', word, 'is not in my vocabulary!')
    return retvec


def createVocablisr(data):
    vocab = set([])
    for word in data:
        vocab = vocab | set(word)
    return list(vocab)


def trainNB0(data, label):
    numDoc = len(data)
    numWords = len(data[0])
    pabusive = label.count(1) / float(numDoc)
    p0num = np.ones(numWords)
    p1num = np.ones(numWords)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numDoc):
        if label[i] == 1:
            p1num += data[i]
            p1denom += 1
        else:
            p0num += data[i]
            p0denom += 1

    p1vec = np.log(p1num / p1denom)
    p0vec = np.log(p0num / p0denom)
    return p0vec, p1vec, pabusive


def classify(target, p0v, p1v, pab):
    p1 = sum(p1v * target) + np.log(pab)
    p0 = sum(p0v * target) + np.log(pab)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


def test():
    data, label = loadData()
    myvocab = createVocablisr(data)
    train_data = []
    for i in range(len(data)):
        train_data.append(setwords(myvocab, data[i]))
    p0v, p1v, pab = trainNB0(train_data, label)
    test_d = ['love', 'my', 'dalmation']
    test_data = np.array(setwords(myvocab, test_d))
    test_data = test_data
    if classify(test_data, p0v, p1v, pab):
        print('侮辱')
    else:
        print('非侮辱类')

    test_d = ['stupid', 'dog', 'mr']
    test_data = np.array(setwords(myvocab, test_d))
    test_data = test_data
    if classify(test_data, p0v, p1v, pab):
        print('侮辱')
    else:
        print('非侮辱类')


if __name__ == '__main__':
    test()
