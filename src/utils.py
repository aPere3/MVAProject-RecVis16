#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains some utilities methods.
"""


import numpy


def labels_to_categoricals(y):
    """
    This method allows to turn labels vectors of shape [n,1], into binary categorical vectors of shape [n, m].
    :param y: a numpy labels vector
    :return: a numpy categoricals vector
    """

    assert isinstance(y, numpy.ndarray)
    assert y.shape[1] == 1

    nb_classes = y.max()
    nb_samples = y.shape[0]
    yout = numpy.zeros([nb_samples, nb_classes+1], dtype=numpy.int8)
    for index in range(0, nb_samples):
        yout[index][y[index]] = 1

    return yout