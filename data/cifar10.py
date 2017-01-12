#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains a method to load cifar10 data. This code is based on the recommendations of the toronto team that
created the set, available at: https://www.cs.toronto.edu/~kriz/cifar.html .
"""

try:
    import cPickle
except:
    import _pickle as cPickle
import numpy
import sys


def load_cifar(dataset="training", path="cifar"):
    """
    The Cifar Loading Method.
    :param dataset: 'training' or 'testing', depending on the data to load
    :param path: path to the cifar files
    :return: X,y: data and labels
    """
    if dataset == "training":
        X = numpy.ndarray([50000, 3072])
        y = numpy.ndarray([50000, 1], dtype=numpy.int8)
        files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        for i in range(0, len(files)):
            data = unpickle(path +"/"+ files[i])
            X[i * 10000:(i + 1) * 10000] = data['data']
            y[i * 10000:(i + 1) * 10000] = numpy.reshape(data['labels'], [-1,1])
    elif dataset == "testing":
        X = numpy.ndarray([10000, 3072])
        y = numpy.ndarray([10000, 1], dtype=numpy.int8)
        files = ['test_batch']
        for i in range(0, len(files)):
            data = unpickle(path + "/"+files[i])
            X[i * 10000:(i + 1) * 10000] = data['data']
            y[i * 10000:(i + 1) * 10000] = numpy.reshape(data['labels'], [-1,1])
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    X = numpy.reshape(X, [-1,3,32,32])
    X = numpy.moveaxis(X,1,3)
    C = numpy.reshape(X,[-1,32*32*3])
    return X, y


def unpickle(file):
    """
    Unpickle method from: https://www.cs.toronto.edu/~kriz/cifar.html
    :param file: path to the file
    :return:
    """
    fo = open(file, 'rb')
    if sys.version_info.major == 2:
        dict = cPickle.load(fo)
    else:
        dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict