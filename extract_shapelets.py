from collections import Counter

import numpy as np

from brute_force import find_shapelets_bf, check_candidate
from genetic import find_shapelets_gen
from particle_swarm import find_shapelets_pso
from ps_ea import find_shapelets_ps_ea
from shapelet_tree import ShapeletTree
from util import subsequence_dist


def extract_shapelet(timeseries, labels, min_len=50, max_len=50, verbose=1):
    """
    Search for the (shapelet, distance) combination that results in the best information gain
    :param timeseries: the timeseries in which the shapelet is sought
    :param labels: the corresponding labels for each of the timeseries
    :param min_len: minimum length of the shapelet
    :param max_len: maximum length of the shapelet
    :param verbose: whether to print intermediary results or not
    :return: a shapelet with corresponding distance that gains the most information
    """
    return find_shapelets_bf(timeseries, labels, max_len=max_len, min_len=min_len,
                              verbose=verbose)


def fit(X, y, max_depth=None, min_samples_split=2, min_len=10, max_len=10):
    if (max_depth is None or max_depth > 0) and len(X) > min_samples_split and len(np.unique(y)) > 1:
        # TODO: pass the distance along with this shapelet so we don't need to recalculate this!
        shapelet = find_shapelets_pso(X, y, max_len=max_len, min_len=min_len)
        distance = check_candidate(X, y, shapelet)[1]
        node = ShapeletTree(right=None, left=None, shapelet=shapelet, distance=distance,
                            class_probabilities=Counter(y))
        X_left, y_left, X_right, y_right = [], [], [], []
        for ts, label in zip(X, y):
            if subsequence_dist(ts, shapelet)[0] <= distance:
                X_left.append(ts)
                y_left.append(label)
            else:
                X_right.append(ts)
                y_right.append(label)

        new_depth = None if max_depth is None else max_depth - 1
        node.left = fit(X_left, y_left, max_depth=new_depth, min_samples_split=min_samples_split,
                        min_len=min_len, max_len=max_len)
        node.right = fit(X_right, y_right, max_depth=new_depth, min_samples_split=min_samples_split,
                         min_len=min_len, max_len=max_len)
        return node
    else:
        return ShapeletTree(right=None, left=None, shapelet=None, distance=None,
                            class_probabilities=Counter(y))


def predict(y):
    pass


def predict_proba(y):
    pass