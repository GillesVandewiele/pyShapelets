# TODO: shapelet transform
# TODO: shapelet tree
# TODO: shapelet isolation tree for anomaly detection


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
import extractors.extractor as extractors 
from util import sdist
import numpy as np
from dtaidistance import dtw
from collections import Counter

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
        print(method)
        self.extractor = {
            'fast': extractors.FastExtractor(),
            'brute': extractors.BruteForceExtractor(),
            'sax': extractors.SAXExtractor(),
            'learn': extractors.LearningExtractor(),
            'genetic': extractors.GeneticExtractor()
        }[method]
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
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = check_array(X)

        feature_vectors = np.zeros((len(X), len(self.shapelets)))
        for smpl_idx, sample in enumerate(X):
            for shap_idx, shapelet in enumerate(self.shapelets):
                feature_vectors[smpl_idx, shap_idx] = sdist(shapelet, sample)

        return feature_vectors


class ShapeletTree(object):
    def __init__(self, right=None, left=None, shapelet=None, threshold=None, class_probabilities={}):
        self.right = right
        self.left = left
        self.shapelet = shapelet
        self.threshold = threshold
        self.class_probabilities = class_probabilities

        def evaluate(self, time_serie, proba=True):
            if self.distance is None:
                if proba:
                    return self.class_probabilities
                else:
                    return max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
            else:
                dist = util.sdist(self.shapelet, time_serie)
                if dist <= self.distance:
                    return self.left.evaluate(time_serie, proba=proba)
                else:
                    return self.right.evaluate(time_serie, proba=proba)

        def predict(self, X):
            return [ evaluate(ts, proba=False) for ts in X ]

        def predict_proba(self, X):
            return [ evaluate(ts, proba=True) for ts in X ]


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
    def __init__(self, method=None, max_depth=None, min_samples_split=1, metric='ig'):
        if method is None:
            method = 'fast'
        self.extractor = {
            'fast': extractors.FastExtractor(),
            'brute': extractors.BruteForceExtractor(),
            'sax': extractors.SAXExtractor(),
            'learn': extractors.LearningExtractor(),
        }[method]

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = set(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    
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