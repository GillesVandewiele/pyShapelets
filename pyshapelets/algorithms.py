# TODO: shapelet transform
# TODO: shapelet tree
# TODO: shapelet isolation tree for anomaly detection


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
import extractors.extractor as extractors 
from util import sdist, sdist_no_norm
import numpy as np
#from dtaidistance import dtw
from collections import Counter
import operator

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util

class ShapeletTransformer(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root..
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """
    def __init__(self, method=None, min_len=None, max_len=None, 
                 nr_shapelets=1, metric='ig'):
        if method is None:
            method = 'fast'

        if type(method) == str:
            self.extractor = {
                'fast': extractors.FastExtractor(),
                'brute': extractors.BruteForceExtractor(),
                'sax': extractors.SAXExtractor(),
                'learn': extractors.LearningExtractor(),
                'genetic': extractors.GeneticExtractor(),
                'pso': extractors.ParticleSwarmExtractor()
            }[method]
        else:
            self.extractor = method
            
        self.shapelets = []
        self.min_len = min_len
        self.max_len = max_len
        self.nr_shapelets = nr_shapelets
        self.metric = metric

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)

        self.input_shape_ = X.shape

        self.shapelets = self.extractor.extract(
            X, y, 
            min_len=self.min_len, 
            max_len=self.max_len, 
            nr_shapelets=self.nr_shapelets, 
            metric=self.metric
        )

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['shapelets'])

        # Input validation
        X = check_array(X)

        feature_vectors = np.zeros((len(X), len(self.shapelets)))
        for smpl_idx, sample in enumerate(X):
            for shap_idx, shapelet in enumerate(self.shapelets):
                feature_vectors[smpl_idx, shap_idx] = util.sdist_sq(shapelet.flatten(), sample)

        return feature_vectors


class ShapeletTree(object):
    def __init__(self, right=None, left=None, shapelet=None, threshold=None, class_probabilities={}):
        self.right = right
        self.left = left
        self.shapelet = shapelet
        self.threshold = threshold
        self.class_probabilities = class_probabilities

    def evaluate(self, time_serie, proba=True):
        if self.is_leaf():
            if proba:
                return self.class_probabilities
            else:
                return max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
        else:
            dist = util.sdist(self.shapelet, time_serie)
            if dist <= self.threshold:
                return self.left.evaluate(time_serie, proba=proba)
            else:
                return self.right.evaluate(time_serie, proba=proba)

    def predict(self, X):
        return [ self.evaluate(ts, proba=False) for ts in X ]

    def predict_proba(self, X):
        return [ self.evaluate(ts, proba=True) for ts in X ]

    def is_leaf(self):
        return self.threshold is None

    def extract_all_shapelets(self):
        if self.is_leaf():
            return None
        else:
            left_shap = self.left.extract_all_shapelets()
            right_shap = self.right.extract_all_shapelets()
            all_shapelets = [self.shapelet]
            if left_shap is not None:
                all_shapelets += left_shap
            if right_shap is not None:
                all_shapelets += right_shap
            return all_shapelets


class ShapeletTreeClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, method=None, max_depth=None, min_samples_split=1, min_len=None, max_len=None, metric='ig'):
        if method is None:
            method = 'fast'

        if type(method) == str:
            self.extractor = {
                'fast': extractors.FastExtractor(),
                'brute': extractors.BruteForceExtractor(),
                'sax': extractors.SAXExtractor(),
                'learn': extractors.LearningExtractor(),
                'genetic': extractors.GeneticExtractor(),
                'pso': extractors.ParticleSwarmExtractor()
            }[method]
        else:
            self.extractor = method

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.metric = metric
        self.min_len = min_len
        self.max_len = max_len

        self.tree = None

    def _calc_probs(self, y):
        probs = Counter(y)
        total = sum(probs.values())
        for k in probs: probs[k] /= total
        return probs


    def _extract_tree(self, X, y, depth=0):
        if (self.max_depth is None or depth > self.max_depth) and len(np.unique(y)) > 1:
            # Extract 1 shapelet, using the specified `extractor`
            map_dict = {}
            for j, c in enumerate(np.unique(y)):
                map_dict[c] = j
            y_mapped = np.vectorize(map_dict.get)(y)

            shapelet = self.extractor.extract(
                X, y_mapped, 
                min_len=self.min_len, 
                max_len=self.max_len, 
                nr_shapelets=1, 
                metric=self.metric
            )[0]

            # Get the best threshold distance for this shapelet
            L = []
            X_left, y_left, X_right, y_right = [], [], [], []
            for k in range(len(X)):
                D = X[k, :]
                dist = util.sdist(shapelet, D)
                L.append((dist, y[k]))
            threshold = util.get_threshold(L)

            # Create a new internal node
            node = ShapeletTree(right=None, left=None, shapelet=shapelet, threshold=threshold, class_probabilities=self._calc_probs(y))

            # Partition the data
            X_left, y_left, X_right, y_right = [], [], [], []
            for ts, label in zip(X, y):
                if util.sdist(shapelet, ts) <= threshold:
                    X_left.append(ts)
                    y_left.append(label)
                else:
                    X_right.append(ts)
                    y_right.append(label)

            X_left = np.array(X_left)
            y_left = np.array(y_left)
            X_right = np.array(X_right)
            y_right = np.array(y_right)

            # Recursive call to create the left and right child of the internal node
            if len(X_left) >= self.min_samples_split:
                node.left = self._extract_tree(X_left, y_left, depth=depth+1)
            else:
                 node.left = ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y_left))
            if len(X_right) >= self.min_samples_split:
                node.right = self._extract_tree(X_right, y_right, depth=depth+1)
            else:
                 node.right = ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y_right))
            return node

        else:
            return ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y))

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.tree = self._extract_tree(X, y)
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['tree'])

        # Input validation
        X = check_array(X)

        return self.tree.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        return self.tree.predict_proba(X)


class ShapeletForestClassifier(BaseEstimator, ClassifierMixin):
    pass

    
class DTWNearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        self.timeseries = []
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.timeseries_ = X
        self.labels_ = y
        self.classes_ = set(y)

    def _get_labels_closest(self, sample):
        distances = [ dtw.distance_fast(sample, ts) for ts in self.timeseries_ ]
        return self.labels_[np.argsort(distances)[:self.n_neighbors]]

    def predict(self, X):
        # Check if fit thas already been called prior to this call
        check_is_fitted(self, ['timeseries_', 'labels_'])
        predictions = []
        for sample in X:
            predictions.append(Counter(self._get_labels_closest(sample)).most_common(1)[0][0])
        return predictions

    def predict_proba(self, X):
        check_is_fitted(self, ['timeseries_', 'labels_'])
        predictions = []
        for sample in X:
            probs = []
            closest_labels = self._get_labels_closest(sample)
            cntr = Counter(closest_labels)
            total = sum(cntr.values())
            for k in cntr: cntr[k] /= total
            for class_ in self.classes_:
                probs.append(cntr[class_])
            predictions.append(probs)
        return predictions