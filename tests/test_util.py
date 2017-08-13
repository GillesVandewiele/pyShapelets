#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
import numpy as np
from pyshapelets.util import util
import pytest
import time

__author__ = "GillesVandewiele"
__copyright__ = "GillesVandewiele"
__license__ = "none"


def test_calculate_stats():
    # Generate two random vectors
    np.random.seed(1337)
    a = np.random.random(100)
    b = np.random.random(100)

    # So calculate_stats is supposed to return 5 arrays:
    # The cumulative sum of a
    # The cumulative sum of a, squared
    # The cumulative sum of b
    # The cumulative sum of b, squared
    # The sum of products of elements in a and b

    # Let's calculate these arrays manually in a slow fashion
    start = time.time()
    s_x = [0]
    s_x_sqr = [0]
    for i in range(1, len(a)+1):
        sum = 0
        sum_sqr = 0
        for x in a[:i]:
            sum += x
            sum_sqr += x**2
        s_x.append(sum)
        s_x_sqr.append(sum_sqr)
    s_x = np.array(s_x)
    s_x_sqr = np.array(s_x_sqr)

    s_y = [0]
    s_y_sqr = [0]
    for i in range(1, len(b)+1):
        sum = 0
        sum_sqr = 0
        for x in b[:i]:
            sum += x
            sum_sqr += x**2
        s_y.append(sum)
        s_y_sqr.append(sum_sqr)
    s_y = np.array(s_y)
    s_y_sqr = np.array(s_y_sqr)

    m_uv = np.zeros((len(a) + 1, len(b) + 1))
    for u in range(len(a)):
        for v in range(len(b)):
            t = abs(u - v)
            if u > v:
                m_uv[u + 1, v + 1] = np.sum([a[i+t]*b[i] for i in range(v+1)])
            else:
                m_uv[u + 1, v + 1] = np.sum([a[i]*b[i+t] for i in range(u+1)])
    print('\nManual method takes', time.time() - start, 'seconds')

    # Let's calculate the arrays with our method
    start = time.time()
    s_x_2, s_x_sqr_2, s_y_2, s_y_sqr_2, m_uv_2 = util.calculate_stats(a, b)
    print('\nMethod with numpy function takes', time.time() - start, 'seconds')

    # Are they 'almost' equal (up to 7 decimals)
    np.testing.assert_almost_equal(s_x, s_x_2)
    np.testing.assert_almost_equal(s_x_sqr, s_x_sqr_2)
    np.testing.assert_almost_equal(s_y, s_y_2)
    np.testing.assert_almost_equal(s_y_sqr, s_y_sqr_2)
    np.testing.assert_almost_equal(m_uv, m_uv_2)


def test_calculate_entropy():
    # Entropy of a pure dataset is 0
    np.testing.assert_almost_equal(util.calculate_entropy([1.]), 0)
    # Maximum entropy for binary case
    np.testing.assert_almost_equal(util.calculate_entropy([0.5, 0.5]), -np.log(0.5))
    np.testing.assert_almost_equal(util.calculate_entropy([1./3., 1./3., 1./3.]), -np.log(1./3.))


def test_calculate_dict_entropy():
    # Pure dataset: entropy should be 0
    np.testing.assert_almost_equal(util.calculate_dict_entropy([0]*100), 0)
    # Worst-case binary dataset: entropy is equal to -(0.5 * log(0.5) + 0.5 * log(0.5))
    np.testing.assert_almost_equal(util.calculate_dict_entropy([0]*50 + [1]*50), -np.log(0.5))
    np.testing.assert_almost_equal(util.calculate_dict_entropy([0]*25 + [1]*25 + [2]*25), -np.log(1./3.))


def test_calculate_information_gain():
    # We first have the worst-case and we do a perfect partitioning (0's on the left, 1's on the right)
    # Our new entropy will be 0, so prior_entropy - 0 = prior_entropy = -log(0.5)
    left_partition = [0]*50
    right_partition = [1]*50
    prior_entropy = -np.log(0.5)
    np.testing.assert_almost_equal(util.information_gain(left_partition, right_partition, prior_entropy), -np.log(0.5))


def test_sdist():
    # Generate random vector, take subseries from it and calculate distance (should be 0 of course)
    np.random.seed(1337)
    a = np.random.random(100)
    b = a[:25]
    # The function should work in both directions
    np.testing.assert_almost_equal(util.sdist(a, b), 0)
    np.testing.assert_almost_equal(util.sdist(b, a), 0)
    b = a[50:]
    np.testing.assert_almost_equal(util.sdist(a, b), 0)


def test_sdist_new():
    a = list(range(5))
    b = a.copy()
    stats = util.calculate_stats(a, b)
    np.testing.assert_almost_equal(util.sdist_new(a, b, 0, stats), 0)


    np.random.seed(1337)
    a = np.random.random(100)
    stats = util.calculate_stats(a, a)
    np.testing.assert_almost_equal(util.sdist_new(a[50:], a, 50, stats), 0)




