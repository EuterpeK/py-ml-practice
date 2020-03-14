from re import sub
from os import listdir
from collections import Counter
from itertools import chain
import numpy as np
from jieba import cut


def getwords(filename):
    words = []
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            line = sub(r'[.【】0-9——。\-，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
        return words


def getVocabList(filedir):
    file_names = listdir(filedir)
    wordlist = set([])
    for name in file_names:
        wordlist = wordlist | set(getwords(filedir + '/' + name))
    return list(wordlist)


def getvec(vocal, data):
    train = [0] * len(vocal)
    for word in data:
        if word in vocal:
            train[vocal.index(word)] += 1
        else:
            print('this word not in the list')

    return train


def getTrain(filename):
    myVocab = getVocabList(filename)
    train = []
    dirs = listdir(filename)
    for name in dirs:
        data = getwords(filename + '/' + name)
        train.append(getvec(myVocab, list(data)))
    return train


def train(train_data, label):
    num_train = len(train_data)
    num_words = len(train_data[0])
    pab = sum(label) / float(num_train)
    p0num = np.ones(num_words)
    p1num = np.ones(num_words)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(num_train):
        if label[i] == 1:
            p1num += train_data[i]
            p1denom += 1
        else:
            p0num += train_data[i]
            p0denom += 1
    p1vec = np.log(p1num / p1denom)
    p0vec = np.log(p0num / p0denom)
    return p0vec, p1vec, pab


def classify(vec, p0vec, p1vec, pab):
    p0 = sum(vec * p0vec) + np.log(pab)
    p1 = sub(vec * p1vec) + np.log(1.0 - pab)
    if p0 > p1:
        return 0
    else:
        return 1


def spamTest():
    vocabList = getVocabList('email/spam')
    train_data = getTrain('email/spam')
    test_data = getTrain('email/ham')
