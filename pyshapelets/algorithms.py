# TODO: shapelet transform
# TODO: shapelet tree
# TODO: shapelet isolation tree for anomaly detection


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import extractors.extractor as extractors 
from util import sdist
import numpy as np

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
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    