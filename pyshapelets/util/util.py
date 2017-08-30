from itertools import combinations, product

import numpy as np
import time
from scipy.stats import entropy
from collections import Counter
import math


def euclidean_distance(a, b, min_dist=np.inf):
    if min_dist == np.inf:
        return np.sum(np.power(a-b, 2))

    dist = 0
    for x, y in zip(a, b):
        dist += (float(x)-float(y))**2
        if dist >= min_dist: return np.inf
    return dist


def calculate_stats(a, b):
    """
    Calculate the five different arrays for two time series. Used for the calculation of normalized euclidean distance
    :param a: timeseries
    :param b: timeseries
    :return: five arrays used for the calculation of normalized euclidean distance
    """
    s_x = np.append([0], np.cumsum(a))
    s_x_sqr = np.append([0], np.cumsum(np.power(a, 2)))
    s_y = np.append([0], np.cumsum(b))
    s_y_sqr = np.append([0], np.cumsum(np.power(b, 2)))

    # s_x = np.cumsum(a)
    # s_x_sqr = np.cumsum(np.power(a, 2))
    # s_y = np.cumsum(b)
    # s_y_sqr = np.cumsum(np.power(b, 2))

    # m_uv = np.zeros((len(a), len(b)))
    # L = max(len(a), len(b))
    # for k in range(L):
    #     if k < len(a):
    #         m_uv[k][0] = a[k] * b[0]
    #         i, j = k + 1, 1
    #         while i < len(a) and j < len(b):
    #             m_uv[i][j] = m_uv[i - 1][j - 1] + a[i] * b[j]
    #             i += 1
    #             j += 1
    #
    #     if k < len(b):
    #         m_uv[0][k] = a[0] * b[k]
    #         i, j = 1, k + 1
    #         while i < len(a) and j < len(b):
    #             m_uv[i][j] = m_uv[i - 1][j - 1] + a[i] * b[j]
    #             i += 1
    #             j += 1
    # m_uv = np.vstack((np.zeros((1, m_uv.shape[1])), m_uv))
    # m_uv = np.hstack((np.zeros((m_uv.shape[0], 1)), m_uv))

    m_uv = np.zeros((len(a) + 1, len(b) + 1))
    for u in range(len(a)):
        for v in range(len(b)):
            t = abs(u-v)
            if u > v:
                m_uv[u+1, v+1] = m_uv[u, v] + a[v+t]*b[v]
            else:
                m_uv[u+1, v+1] = m_uv[u, v] + a[u]*b[u+t]
    # m_uv = m_uv[1:, 1:]
    return s_x, s_x_sqr, s_y, s_y_sqr, m_uv


def calculate_stats_old(a, b):
    s_x = np.append([0], np.cumsum(a))
    s_x_sqr = np.append([0], np.cumsum(np.power(a, 2)))
    s_y = np.append([0], np.cumsum(b))
    s_y_sqr = np.append([0], np.cumsum(np.power(b, 2)))
    m_uv = np.zeros((len(a) + 1, len(b) + 1))
    start = time.time()
    for u in range(len(a)):
        for v in range(len(b)):
            t = abs(u - v)
            if u > v:
                m_uv[u + 1, v + 1] = np.sum([a[i+t]*b[i] for i in range(v+1)])
            else:
                m_uv[u + 1, v + 1] = np.sum([a[i]*b[i+t] for i in range(u+1)])
    return s_x, s_x_sqr, s_y, s_y_sqr, m_uv


def normalized_euclidean_distance(a, b, stats):
    """
    Calculate the normalized (z-normalization and length normalization) euclidean distance between a sub- and timeseries
    # Reference 1: http://www.cs.ucr.edu/~eamonn/LogicalShapelet.pdf
    # Reference 2: http://making.csie.ndhu.edu.tw/seminar/making/papers/PDF/Fast%20Approximate%20Correlation%20for%20Massive%20Time-series%20Data.pdf
    :param a: list, the subserie (shapelet)
    :param b: list, the timeseries
    :return: the normalized euclidean distance
    """
    s_x, s_x_sqr, s_y, s_y_sqr, m_uv = stats
    min_sum = np.inf
    # Convolute the subserie over the longer timeseries to get the minimal distance
    for v in range(len(b) - len(a)):
        mu_x = s_x[-1] / len(a)   # The final element of s_x contains the sum of all elements in x
        sigma_x = s_x_sqr[-1] / len(a) - mu_x ** 2
        mu_y = (s_y[v+len(a)] - s_y[v]) / len(a)   # Grab subserie of length len(a) from b and calculate mean
        sigma_y = (s_y_sqr[v+len(a)] - s_y_sqr[v]) / len(a) - mu_y ** 2
        m = m_uv[len(a), v+len(a)] - m_uv[0, v]

        dist = (m - len(a) * mu_x * mu_y) / (m*sigma_x*sigma_y)
        min_sum = min(min_sum, dist)

    return np.sqrt(min_sum)


def calculate_entropy(probabilities):
    # This takes logarithm with base 10 instead of 2 (should not be a problem)
    return entropy(probabilities)
    #return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def calculate_dict_entropy(data):
    cntr = Counter(data)
    total = sum(cntr.values())
    for key in cntr:
        cntr[key] /= total
    return calculate_entropy(list(cntr.values()))


def information_gain(labels_left, labels_right, prior_entropy):
    total_length = len(labels_left) + len(labels_right)
    if total_length == 0:
        print('break')
    ig_left = float(len(labels_left)) / total_length * calculate_dict_entropy(labels_left)
    ig_right = float(len(labels_right)) / total_length * calculate_dict_entropy(labels_right)
    return prior_entropy - (ig_left + ig_right)


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) <= len(time_serie):
        min_dist, min_idx = float("inf"), -1
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = euclidean_distance(sub_serie, time_serie[i:i + len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return subsequence_dist(sub_serie, time_serie)


def subsequence_dist_z_space(time_serie, sub_serie, stats):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), -1
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = normalized_euclidean_distance(sub_serie, time_serie[i:i + len(sub_serie)], stats)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


def znorm(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return (x - np.mean(x))/np.std(x)


def sdist(x, y):
    if len(x) > len(y):
        return sdist(y, x)
    else:
        min_dist = np.inf
        x = znorm(x)
        for j in range(len(y) - len(x) + 1):
            z = znorm(y[j:j+len(x)])
            dist = np.linalg.norm(z-x, ord=2)
            if dist < min_dist:
                min_dist = dist

        return min_dist/len(x)


def sdist_new(x, y, i, stats):
    # i = start_position in time serie where x is from
    l = len(x)
    if l > len(y):
        return sdist_new(y, x, i, stats)
    elif l == 1:
        return np.min(np.abs(np.add(y, -x)))  # Return the value closest to x
    else:
        min_dist = np.inf
        mu_x = (stats[0][i + l] - stats[0][i]) / l
        sigma_x = (stats[1][i + l] - stats[1][i]) / l - mu_x ** 2
        for v in range(len(y) - l + 1):
            xy = stats[4][i + l, v + l] - stats[4][i, v]
            mu_y = (stats[2][v + l] - stats[2][v]) / l
            sigma_y = (stats[3][v + l] - stats[3][v]) / l - mu_y ** 2

            C = (xy - l*mu_x*mu_y) / (l * math.sqrt(sigma_x * sigma_y))

            if C <= 1:
                dist = 2*l*(1 - C)
                min_dist = min(dist, min_dist)
            # else:
            #     min_dist = 0

        return math.sqrt(min_dist)


def upperIG(L, R, timeseries, labels):
    """ Calculate an upper bound on the information gain for a new candidate
    :param L: sorted array, containing distances of the timeseries of the dataset to shapelet candidate
    :param R: a new candidate distance R
    :param timeseries: the timeseries of the dataset
    :param labels: the labels of the dataset
    :return: an upper bound on IG
    """
    max_ig = 0
    cntr = 0
    prior_entropy = calculate_dict_entropy(labels)
    class_mapping = {}
    unique_classes = np.unique(labels)
    for _class in unique_classes:
        class_mapping[_class] = cntr
        cntr += 1

    for k in range(len(timeseries)-1):
        tau = (L[k][0] + L[k+1][0]) / 2
        C = len(np.unique(labels))
        p = 0

        # Partition the points, except for the ones in [tau - R, tau + R]
        left_partition, right_partition = [], []
        points_to_consider = []
        while L[p][0] < tau - R:
            left_partition.append(L[p][1])
            p += 1
        left_partition_counter = Counter(left_partition)

        while p < len(L) and L[p][0] < tau + R:
            points_to_consider.append(p)
            p += 1
        while p < len(L):
            right_partition.append(L[p][1])
            p += 1
        right_partition_counter = Counter(right_partition)

        for transfer_direction in list(product([0,1], repeat=C)):
            temp_right_partition = []
            temp_left_partition = []
            for p in points_to_consider:
                if transfer_direction[class_mapping[L[p][1]]]:
                    # Move the point to its majority end
                    if left_partition_counter[L[p][1]] > right_partition_counter[L[p][1]]:
                        temp_left_partition.append(L[p][1])
                    else:
                        temp_right_partition.append(L[p][1])
                else:
                    if L[p][0] <= tau:
                        temp_left_partition.append(L[p][1])
                    else:
                        temp_right_partition.append(L[p][1])

            ig = information_gain(left_partition + temp_left_partition, right_partition + temp_right_partition,
                                  prior_entropy)
            max_ig = max(ig, max_ig)

    return max_ig


def partitions(items, k):
    def split(indices):
        i=0
        for j in indices:
            yield items[i:j]
            i = j
        yield items[i:]

    for indices in combinations(range(1, len(items)), k-1):
        yield list(split(indices))


def information_gain_ub(labels):
    # TODO: Can we find a mathematical formula for this?
    cntr = Counter(labels)
    all_values = list(cntr.values())
    all_sum = sum(all_values)
    for i in range(len(all_values)): all_values[i] /= all_sum
    prior_entropy = calculate_entropy(all_values)
    best_ig = 0

    for left, right in partitions(list(cntr.values()), 2):
        left_sum = sum(left)
        right_sum = sum(right)
        for i in range(len(left)): left[i] /= left_sum
        for i in range(len(right)): right[i] /= right_sum
        ig = information_gain(left, right, prior_entropy)
        if ig > best_ig: best_ig = ig

    return best_ig