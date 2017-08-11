import operator
import numpy as np
from util import subsequence_dist


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
            dist = subsequence_dist(time_serie, self.shapelet)
            if dist < self.distance:
                return self.left.evaluate(time_serie, proba=proba)
            else:
                return self.right.evaluate(time_serie, proba=proba)

    def predict(self, time_series):
        results = np.zeros((len(time_series), 1))
        for i, time_serie in enumerate(time_series):
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
