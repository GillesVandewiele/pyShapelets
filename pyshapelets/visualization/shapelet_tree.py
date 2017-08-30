import operator
import numpy as np

from pyshapelets.shapelet_extraction.brute_force import check_candidate
from pyshapelets.util.util import subsequence_dist, subsequence_dist_z_space, sdist, calculate_stats, sdist_new


class ShapeletTree(object):

    def __init__(self, right=None, left=None, shapelet=None, distance=None, class_probabilities={}):
        self.right = right
        self.left = left
        self.shapelet = shapelet
        self.distance = distance
        self.class_probabilities = class_probabilities

    def evaluate(self, time_serie, proba=True):
        if self.distance is None:
            if proba:
                return self.class_probabilities
            else:
                return max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
        else:
            dist, idx = subsequence_dist(time_serie, self.shapelet)
            if dist <= self.distance:
                return self.left.evaluate(time_serie, proba=proba)
            else:
                return self.right.evaluate(time_serie, proba=proba)

    def evaluate_z_norm_space(self, time_serie, proba=True):
        if self.distance is None:
            if proba:
                return self.class_probabilities
            else:
                return max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
        else:
            stats = calculate_stats(self.shapelet, time_serie)
            dist = sdist_new(self.shapelet, time_serie, 0, stats)
            if dist <= self.distance:
                return self.left.evaluate_z_norm_space(time_serie, proba=proba)
            else:
                return self.right.evaluate_z_norm_space(time_serie, proba=proba)

    def predict(self, time_series, z_norm=False):
        results = np.zeros((len(time_series), 1))
        for i, time_serie in enumerate(time_series):
            if z_norm:
                results[i] = self.evaluate_z_norm_space(time_serie, proba=False)
            else:
                results[i] = self.evaluate(time_serie, proba=False)
        return results

    def predict_proba(self, time_serie):
        return self.evaluate(time_serie, proba=True)

    def get_depth(self, depth=0):
        left_depth, right_depth = 0, 0
        if self.left is not None:
            left_depth = self.left.get_depth(depth=depth+1)
        if self.right is not None:
            right_depth = self.right.get_depth(depth=depth+1)
        return max(left_depth, right_depth, depth)

    def _reset_class_probs(self):
        self.class_probabilities = {}
        if self.left is not None:
            self.left._reset_class_probs()
        if self.right is not None:
            self.right._reset_class_probs()

    def increment_class_probs(self, ts, label):
        if label not in self.class_probabilities:
            self.class_probabilities[label] = 1
        else:
            self.class_probabilities[label] += 1

        if self.distance is not None:

            dist, idx = subsequence_dist(ts, self.shapelet)
            if dist <= self.distance:
                self.left.increment_class_probs(ts, label)
            else:
                self.right.increment_class_probs(ts, label)

    def populate_class_probs(self, timeseries, labels):
        # Put the class probabilities of all nodes (including internal nodes) on all 0's
        self._reset_class_probs()

        # For each timeserie, run through the tree and increment the class probs, according to corresponding label
        for (ts, label) in zip(timeseries, labels): self.increment_class_probs(ts, label)

    def recalculate_distances(self, timeseries, labels):
        if self.distance is not None:
            ig, dist = check_candidate(timeseries, labels, self.shapelet)
            print(dist, self.distance)
            self.distance = dist
            ts_left, labels_left = [], []
            ts_right, labels_right = [], []
            for (ts, label) in zip(timeseries, labels):
                dist, idx = subsequence_dist(ts, self.shapelet)
                if dist < self.distance:
                    ts_left.append(ts)
                    labels_left.append(label)
                else:
                    ts_right.append(ts)
                    labels_right.append(label)

            print(labels, 'are split into', labels_left, 'and', labels_right)

            self.left.recalculate_distances(ts_left, labels_left)
            self.right.recalculate_distances(ts_right, labels_right)

        else:
            print('leaf:', labels)
