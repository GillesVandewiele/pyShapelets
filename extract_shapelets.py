import numpy as np

from brute_force import find_shapelets_bf
from genetic import find_shapelets_gen
from particle_swarm import find_shapelets_pso
from ps_ea import find_shapelets_ps_ea


def extract_shapelet(timeseries, labels, min_len=10, max_len=15, verbose=1):
    """
    Search for the (shapelet, distance) combination that results in the best information gain
    :param timeseries: the timeseries in which the shapelet is sought
    :param labels: the corresponding labels for each of the timeseries
    :param min_len: minimum length of the shapelet
    :param max_len: maximum length of the shapelet
    :param verbose: whether to print intermediary results or not
    :return: a shapelet with corresponding distance that gains the most information
    """
    return find_shapelets_ps_ea(timeseries, labels, max_len=max_len, min_len=min_len,
                                verbose=verbose)


def fit(X, y, max_depth=None, min_samples_split=2, min_len=10, max_len=15):
    if (max_depth is None or max_depth > 0) and len(X) > min_samples_split:
        # TODO: extract shapelet, split the data in two and call recursively (max_depth--)
        pass
    else:
        # TODO: create a leaf for the tree
        pass


def predict(y):
    pass


def predict_proba(y):
    pass