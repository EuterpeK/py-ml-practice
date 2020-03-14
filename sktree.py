from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# import pydotplus
# from sklearn.externals.six import StringIO

'''
if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)

    lenseLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses, lenseLables)
'''

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr]
    lense_target = [each[-1] for each in lenses]
    # for each in lenses:
    # lense_target.append(each[-1])

    lenseLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lenseLables:
        for each in lenses:
            lenses_list.append(each[lenseLables.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)

    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
