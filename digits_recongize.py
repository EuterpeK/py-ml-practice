from sklearn.neighbors import KNeighborsClassifier as kNN
import numpy as np
from os import listdir


def ing2vector(filename):
    vec = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                vec[0, i * 32 + j] = int(line[j])

    return vec


def test():
    labels = []
    file_names = listdir('trainingDigits')
    datas = np.zeros((len(file_names), 1024))
    for i in range(len(file_names)):
        name = file_names[i]
        label = name.split('_')[0]
        labels.append(label)
        data = ing2vector('trainingDigits/' + name)
        datas[i, :] = data[0, :]

    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(datas, labels)

    test_name = listdir('testDigits')
    error = 0
    for i in range(len(test_name)):
        name = test_name[i]
        label = name.split('_')[0]
        vect = ing2vector('testDigits/' + name)
        pre_result = neigh.predict(vect)
        if pre_result != label:
            error += 1
        print("real:", label, 'predict:', pre_result)

    print('tot_error:', error, 'error_rate:', error / len(test_name))


if __name__ == '__main__':
    test()
