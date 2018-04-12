import numpy as np
from scipy.stats import zscore, pearsonr, entropy
import time
from collections import Counter
from sklearn.feature_selection import mutual_info_classif


def best_ig(L, max_gain, max_gap):
	# TODO: can we do this more efficiently and more clean?
	classes = set([x[1] for x in L])
	max_tau = (L[0][0] + L[1][0]) / 2
	updated = False
	for k in range(len(L) - 1):
		tau = (L[k][0] + L[k + 1][0]) / 2
		left_labels = [x[1] for x in L[:k+1]]
		right_labels = [x[1] for x in L[k+1:]]
		all_labels = left_labels + right_labels
		left_counts = []
		right_counts = []
		all_counts = []
		for c in classes:
			left_counts.append(left_labels.count(c))
			right_counts.append(right_labels.count(c))
			all_counts.append(all_labels.count(c))
		left_entropy = len(left_labels)/len(L) * entropy(left_counts)
		right_entropy = len(right_labels)/len(L) * entropy(right_counts)
		ig = entropy(all_counts) - left_entropy - right_entropy
		g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
		if ig > max_gain or (ig == max_gain and g > max_gap):
			max_tau = tau
			max_gain = ig
			max_gap = g
			updated = True

	return max_tau, updated, max_gain, max_gap


def best_ig_sklearn(L, max_gain):
	X = np.reshape([x[0] for x in L], (-1, 1))
	y = [x[1] for x in L]
	ig = mutual_info_classif(X, y)
	updated = ig > max_gain
	return ig


def test_information_gains():
	X = np.random.rand(250)
	y = np.random.randint(5, size=250)
	L = list(zip(X, y))
	print(best_ig(L, 0, 0))
	print(best_ig_sklearn(L, 0))


#test_information_gains()

def z_norm(x):
    """Normalize time series such that it has zero mean and unit variance"""
    # IMPORTANT: faster than scipy.stats.zscore for smaller arrays
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x == 0: sigma_x = 1
    return (x - mu_x) / sigma_x


def norm_euclidean_distance(x, y):
    """Calculate the length-normalized euclidean distance."""
    return 1/np.sqrt(len(x)) * np.linalg.norm(x - y)


def sdist(x, y):
	if len(y) < len(x): return sdist(y, x)
	min_dist = np.inf
	norm_x = z_norm(x)
	for j in range(len(y) - len(x) + 1):
		norm_y = z_norm(y[j:j+len(x)])
		dist = norm_euclidean_distance(norm_x, norm_y)
		if dist < min_dist:
			min_dist = dist
	return min_dist

def pearson(x, y):
    """Calculate the correlation between two time series"""
    # IMPORTANT: always faster than scipy.stats.pearsonr and np.corrcoeff
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    m = len(x)
    return (np.sum(x * y) - (m * mu_x * mu_y)) / (m * sigma_x * sigma_y)


def pearson_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M):
    """Calculate the correlation between two time series. Calculate
    the mean and standard deviations by using the statistic arrays."""
    mu_x = (S_x[u + l] - S_x[u]) / l
    mu_y = (S_y[v + l] - S_y[v]) / l
    sigma_x = np.sqrt((S_x2[u + l] - S_x2[u]) / l - mu_x ** 2)
    sigma_y = np.sqrt((S_y2[v + l] - S_y2[v]) / l - mu_y ** 2)
    xy = M[u + l, v + l] - M[u, v]
    return (xy - (l * mu_x * mu_y)) / (l * sigma_x * sigma_y)


def calculate_metric_arrays(x, y):
    """Calculate five statistic arrays:
    	* S_x:  contains the cumulative sum of elements of x
		* S_x2: contains the cumulative sum of squared elements of x
    	* S_y:  contains the cumulative sum of elements of y
		* S_y2: contains the cumulative sum of squared elements of y
		* M:    stores the sum of products of subsequences of x and y
    """
    S_x = np.append([0], np.cumsum(x))
    S_x2 = np.append([0], np.cumsum(np.power(x, 2)))
    S_y = np.append([0], np.cumsum(y))
    S_y2 = np.append([0], np.cumsum(np.power(y, 2)))

    # TODO: can we calculate M more efficiently (numpy or scipy)??
    M = np.zeros((len(x) + 1, len(y) + 1))
    for u in range(len(x)):
        for v in range(len(y)):
            t = abs(u-v)
            if u > v:
                M[u+1, v+1] = M[u, v] + x[v+t]*y[v]
            else:
                M[u+1, v+1] = M[u, v] + x[u]*y[u+t]

    return S_x, S_x2, S_y, S_y2, M


def pearson_dist(x, y):
    """Calculate the normalized euclidean distance based on the pearson
    correlation. References:
        1) Rafiei, Davood, and Alberto Mendelzon. "Similarity-based queries for 
           time series data." ACM SIGMOD Record. Vol. 26. No. 2. ACM, 1997.
        2) Mueen, Abdullah, Suman Nath, and Jie Liu. "Fast approximate 
           correlation for massive time-series data." Proceedings of the 2010 
           ACM SIGMOD International Conference on Management of data. ACM, 2010.
    """
    return np.sqrt(2 * (1 - pearson(x, y)))


def pearson_dist_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M):
    return np.sqrt(2 * (1 - pearson_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M)))


def test_distance_metrics():
    np.random.seed(1337)

    # Test 1: check if the euclidean distance is working correctly. 
    x = np.array([1]*10)
    y = np.array([0]*10)

    np.testing.assert_equal(norm_euclidean_distance(x, y), 1.0)

    # Test 2: check if the z-normalization is working properly
    x = np.random.normal(size=2500000)
    y = np.random.normal(size=2500000)
    np.testing.assert_almost_equal(x, z_norm(x), decimal=2)

    # Test 3: check if the normalized euclidean distance is indeed equal
    # to the formula given in `pearson_dist`
    x = np.random.rand(10)
    y = np.random.rand(10)
    np.testing.assert_almost_equal(
        norm_euclidean_distance(z_norm(x), z_norm(y)),
        pearson_dist(x, y)
    )

    # Test 4: check if the metrics are calculated correctly
    x = np.random.randint(100, size=250)
    y = np.random.randint(100, size=250)
    S_x, S_x2, S_y, S_y2, M = calculate_metric_arrays(x, y)
    np.testing.assert_almost_equal(
        pearson(x, y), 
        pearson_metrics(0, 0, len(x), S_x, S_x2, S_y, S_y2, M)
    )
    np.testing.assert_almost_equal(
        pearson_dist(x, y), 
        pearson_dist_metrics(0, 0, len(x), S_x, S_x2, S_y, S_y2, M)
    )
    np.testing.assert_almost_equal(
        pearson(x[125:], y[125:]), 
        pearson_metrics(125, 125, 125, S_x, S_x2, S_y, S_y2, M)
    )
    np.testing.assert_almost_equal(
        pearson_dist(x[125:], y[125:]), 
        pearson_dist_metrics(125, 125, 125, S_x, S_x2, S_y, S_y2, M)
    )