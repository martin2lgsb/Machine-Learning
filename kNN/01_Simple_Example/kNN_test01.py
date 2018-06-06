# -*- coding: UTF-8 -*-
import numpy as np
import collections


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ["Love", "Love", "Action", "Action"]
    return group, labels


def classify(inx, dataset, labels, k):
    dist = np.sum((inx - dataset)**2, axis=1)**2
    k_labels = [labels[index] for index in dist.argsort()[0 : k]]
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


if __name__ == '__main__':
    group, labels = createDataSet()
    test = [101, 20]
    test_class = classify(test, group, labels, 3)
    print(test_class)