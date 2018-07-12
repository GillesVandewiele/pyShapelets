import numpy as np
import pandas as pd
from scipy.stats import (zscore, pearsonr, entropy,
                         kruskal, f_oneway, median_test)
import time
from collections import Counter, defaultdict
from sklearn.feature_selection import mutual_info_classif
import math

from sklearn.neighbors import KDTree, BallTree
from scipy.spatial.distance import pdist, euclidean

def separability_index_kd(X, y):
    kdt = KDTree(X, leaf_size=10, metric='euclidean')
    nearest_neighbors = kdt.query(X, k=2, return_distance=False)[:, 1]
    matches = 0
    for i in range(len(X)):
        matches += (y[i] == y[nearest_neighbors[i]])
    return matches / len(X)


def separability_index_ball(X, y):
    bt = BallTree(X, leaf_size=50, metric='euclidean')
    nearest_neighbors = bt.query(X, k=2, return_distance=False)[:, 1]
    matches = 0
    for i in range(len(X)):
        matches += (y[i] == y[nearest_neighbors[i]])
    return matches / len(X)

def class_scatter_matrix(X, y):
    # Works faster than Linear Regression and correlates well with predictive performance (e.g. accuracy)
    # FROM: https://datascience.stackexchange.com/questions/11554/varying-results-when-calculating-scatter-matrices-for-lda
    # Construct a mean vector per class
    mean_vecs = {}
    for label in set(y):
        mean_vecs[label] = np.mean(X[y==label], axis=0)
        
    # Construct the within class matrix (S_w)
    d = X.shape[1]
    S_w = np.zeros((d, d))
    for label, mv in zip(set(y), mean_vecs):
        class_scatter = np.cov(X[y==label].T)
        S_w += class_scatter
        
    # Construct an overall mean vector
    mean_overall = np.mean(X, axis=0)
    
    # Construct the between class matrix (S_b)
    S_b = np.zeros((d, d))
    for i in mean_vecs:
        mean_vec = mean_vecs[i]
        n = X[y==i, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        S_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        
    return np.trace(S_b) / np.trace(S_w + S_b)

def  big_s(x, center):
    len_x = len(x)
    total = 0
        
    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)    
    
    return total/len_x

def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0    

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res

def dbi(X, labels):
    n_cluster = len(set(labels))
    cluster_k = [X[labels == k] for k in set(labels)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    dbi = davisbouldin(cluster_k, centroids)
    if not pd.isnull(dbi):
        return dbi
    else:
        return np.inf

def bhattacharyya(X, y, cells_per_dim=10):
    # Calculate lower and upper bound for each dimension
    bounds = {}
    widths = {}
    col_to_del = []
    cntr = 0
    for d in range(X.shape[1]):
        mi, ma = min(X[:, d]), max(X[:, d])
        if (ma - mi) < 1e-5:
            col_to_del.append(d)
        else:
            bounds[cntr] = (mi, ma)
            widths[cntr] = (ma - mi) / cells_per_dim
            cntr += 1
    
    X = np.delete(X, col_to_del, axis=1)
    #print(X, col_to_del)

    if X.shape[1] == 0:
        return 1
            
    # For each datapoint, calculate its cell in the hypercube
    cell_assignment_counts = []
    label_to_idx = {}
    for i, l in enumerate(set(y)): 
        cell_assignment_counts.append(defaultdict(int))
        label_to_idx[l] = i
        
    unique_assignments = set()
    for point_idx, l in zip(range(X.shape[0]), y):
        assignment = []
        for dim_idx in range(X.shape[1]):
            val = X[point_idx, dim_idx]
            cell = (val - bounds[dim_idx][0])//widths[dim_idx]
            assignment.append(cell)
        cell_assignment_counts[label_to_idx[l]][tuple(assignment)] += 1
        unique_assignments.add(tuple(assignment))
        
    totals = {}
    for l in set(y):
        totals[l] = sum(cell_assignment_counts[label_to_idx[l]].values())
    
    dist = 0
    for assign in unique_assignments:
        temp = 1
        for l in set(y):
            temp *= cell_assignment_counts[label_to_idx[l]][assign] / totals[l]
        dist += (temp * (temp != 1)) ** (1 / len(totals))
    
    return dist

def class_scatter_measure(X, y):
    pass

def direct_class_seperability_measure(X, y):
    pass

def kruskal_score(L):
    score = kruskal(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def f_score(L):
    score = f_oneway(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def mood_median(L):
    score = median_test(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def get_distances_per_class(L):
    distances_per_class = defaultdict(list)
    for dist, label in L:
        distances_per_class[label].append(dist)
    return distances_per_class


def information_gain(prior_entropy, left_counts, right_counts):
    N_left = sum(left_counts)
    N_right = sum(right_counts)
    N = N_left + N_right
    left_entropy = N_left/N * entropy(left_counts)
    right_entropy = N_right/N * entropy(right_counts)
    return prior_entropy - left_entropy - right_entropy


def calculate_ig(L):
    L = sorted(L, key=lambda x: x[0])
    all_labels = [x[1] for x in L]
    classes = set(all_labels)

    left_counts, right_counts, all_counts = {}, {}, {}
    for c in classes: all_counts[c] = 0

    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))

    max_tau = (L[0][0] + L[1][0]) / 2
    max_gain, max_gap = float('-inf'), float('-inf')
    updated = False
    for k in range(len(L) - 1):
        for c in classes: 
            left_counts[c] = 0
            right_counts[c] = 0

        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2
        
        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        for label in left_labels: left_counts[label] += 1
        for label in right_labels: right_counts[label] += 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
        
        if ig > max_gain or (ig == max_gain and g > max_gap):
            max_tau, max_gain, max_gap = tau, ig, g

    return (max_gain, max_gap)

def get_threshold(L):
    L = sorted(L, key=lambda x: x[0])
    all_labels = [x[1] for x in L]
    classes = set(all_labels)

    left_counts, right_counts, all_counts = {}, {}, {}
    for c in classes: all_counts[c] = 0

    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))

    max_tau = (L[0][0] + L[1][0]) / 2
    max_gain, max_gap = float('-inf'), float('-inf')
    updated = False
    for k in range(len(L) - 1):
        for c in classes: 
            left_counts[c] = 0
            right_counts[c] = 0

        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2
        
        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        for label in left_labels: left_counts[label] += 1
        for label in right_labels: right_counts[label] += 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
        
        if ig > max_gain or (ig == max_gain and g > max_gap):
            max_tau, max_gain, max_gap = tau, ig, g

    return max_tau

def upper_ig(L, R):
    # IMPORTANT: for the multi-class case this is not an exact bound
    # but this would require an extra loop of 2**C iterations, C being the 
    # number of classes
    max_ig = 0
    all_labels = [x[1] for x in L]
    classes = set(all_labels)
    
    all_counts = {}
    for c in classes: all_counts[c] = 0
    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))
    
    for k in range(len(L) - 1):
        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2

        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        left_counts, right_counts = {}, {}
        for c in classes:
            left_counts[c] = 0
            right_counts[c] = 0

        for label in left_labels:
            left_counts[label] += 1
        for label in right_labels:
            right_counts[label] += 1

        major_ends = {}
        for c in classes:
            left_fraction = left_counts[c] / len(left_labels)
            right_fraction = right_counts[c] / len(right_labels)
            if left_fraction > right_fraction:
                major_ends[c] = -1
            elif right_fraction > left_fraction:
                major_ends[c] = 1
            else:
                major_ends[c] = 0

        for i in range(k, -1, -1):
            if L[i][0] < tau - R:
                break

            if major_ends[L[i][1]] == 1:
                left_counts[L[i][1]] -= 1
                right_counts[L[i][1]] += 1

        for i in range(k + 1, len(all_labels)):
            if L[i][0] > tau + R:
                break

            if major_ends[L[i][1]] == -1:
                left_counts[L[i][1]] += 1
                right_counts[L[i][1]] -= 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        max_ig = max(ig, max_ig)    

    return max_ig    


def best_ig(L, max_gain, max_gap):
    # TODO: more efficiently and more clean? (mutual_info_classif???)
    # TODO: take a look at numpy hist2d!!!
    
    all_labels = [x[1] for x in L]
    classes = set(all_labels)

    left_counts, right_counts, all_counts = {}, {}, {}
    for c in classes: all_counts[c] = 0

    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))

    max_tau = (L[0][0] + L[1][0]) / 2
    updated = False
    for k in range(len(L) - 1):
        for c in classes: 
            left_counts[c] = 0
            right_counts[c] = 0

        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2
        
        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        for label in left_labels: left_counts[label] += 1
        for label in right_labels: right_counts[label] += 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
        
        if ig > max_gain or (ig == max_gain and g > max_gap):
            max_tau, max_gain, max_gap, updated = tau, ig, g, True

    return max_tau, updated, max_gain, max_gap


def z_norm(x):
    """Normalize time series such that it has zero mean and unit variance"""
    # IMPORTANT: faster than scipy.stats.zscore for smaller arrays (< 1 mill)
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x == 0: sigma_x = 1
    return (x - mu_x) / sigma_x


def norm_euclidean_distance(x, y):
    """Calculate the length-normalized euclidean distance."""
    return 1/np.sqrt(len(x)) * np.linalg.norm(x - y)


def local_square_dist(x, y):
    print(x.shape, y.shape)
    x_sq = np.sum(x ** 2, axis=1)
    y_sq = np.reshape(y ** 2, (1, 1))
    xy = np.dot(x, y)
    print(x_sq.shape, y_sq.shape, xy.shape, y, y_sq)
    input()
    return np.min((x_sq + y_sq - 2 * xy) / len(y))


def sdist(x, y):
    if len(y) < len(x): return sdist(y, x)
    min_dist = np.inf
    norm_x = z_norm(x)
    for j in range(len(y) - len(x) + 1):
        norm_y = z_norm(y[j:j+len(x)])
        dist = norm_euclidean_distance(norm_x, norm_y)
        min_dist = min(dist, min_dist)
    return min_dist


def sdist_no_norm(x, y):
    if len(y) < len(x): return sdist_no_norm(y, x)
    min_dist = np.inf
    for j in range(len(y) - len(x) + 1):
        dist = euclidean(x, y[j:j+len(x)])
        min_dist = min(dist, min_dist)
    return min_dist #/ np.sqrt(len(x))

def sdist_sq(x, y):
    if len(y) < len(x): return sdist_sq(y, x)
    min_dist = np.inf
    for j in range(len(y) - len(x) + 1):
        dist = x**2 + y[j:j+len(x)]**2 - 2*x*y[j:j+len(x)]
        min_dist = min(dist, min_dist)
    return min_dist


def sdist_with_pos(x, y):
    if len(y) < len(x): return sdist(y, x)
    min_dist = np.inf
    norm_x = z_norm(x)
    best_pos = 0
    for j in range(len(y) - len(x) + 1):
        norm_y = z_norm(y[j:j+len(x)])
        dist = norm_euclidean_distance(norm_x, norm_y)
        if dist < min_dist:
        	min_dist = dist
        	best_pos = j
    return min_dist, best_pos


def pearson(x, y):
    """Calculate the correlation between two time series"""
    # IMPORTANT: always faster than scipy.stats.pearsonr and np.corrcoeff
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    m = len(x)
    if sigma_x == 0: sigma_x = 1
    if sigma_y == 0: sigma_y = 1
    return (np.sum(x * y) - (m * mu_x * mu_y)) / (m * sigma_x * sigma_y)


def pearson_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M):
    """Calculate the correlation between two time series. Calculate
    the mean and standard deviations by using the statistic arrays."""
    mu_x = (S_x[u + l] - S_x[u]) / l
    mu_y = (S_y[v + l] - S_y[v]) / l
    sigma_x = np.sqrt((S_x2[u + l] - S_x2[u]) / l - mu_x ** 2)
    sigma_y = np.sqrt((S_y2[v + l] - S_y2[v]) / l - mu_y ** 2)
    xy = M[u + l, v + l] - M[u, v]
    if sigma_x == 0 or pd.isnull(sigma_x): sigma_x = 1
    if sigma_y == 0 or pd.isnull(sigma_y): sigma_y = 1
    return min(1, (xy - (l * mu_x * mu_y)) / (l * sigma_x * sigma_y))


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


def sdist_metrics(u, l, S_x, S_x2, S_y, S_y2, M):
    min_dist = np.inf
    for v in range(len(S_y) - l):
        dist = pearson_dist_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M)
        min_dist = min(dist, min_dist)
    return min_dist


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


def test_quality_metrics():
    # First, we create an order line which is able to achieve
    # perfect separation. Then, we create an order line in which
    # it is impossible to achieve good separation. We also create an
    # order line that is between these two extreme cases.
    good_L  = [
        (0.0, 0), (0.1, 0), (0.15, 0), (0.3, 0), (0.45, 0),
        (0.55, 1), (0.7, 1), (0.75, 1), (0.9, 1), (0.95, 1)
    ]
    avg_L  = [
        (0.0, 0), (0.1, 0), (0.15, 0), (0.3, 0), (0.45, 1),
        (0.55, 0), (0.7, 1), (0.75, 1), (0.9, 1), (0.95, 1)
    ]
    bad_L  = [
        (0.0, 0), (0.1, 1), (0.15, 0), (0.3, 1), (0.45, 0),
        (0.55, 1), (0.7, 0), (0.75, 1), (0.9, 0), (0.95, 1)
    ]

    # The quality metric of the good order line should always be
    # high than that of the average and the metric of the average line
    # should be higher than the metric of the bad order line.
    good_ig = calculate_ig(good_L)[0]
    avg_ig = calculate_ig(avg_L)[0]
    bad_ig = calculate_ig(bad_L)[0]
    assert good_ig > avg_ig > bad_ig

    good_kw = kruskal_score(good_L)
    avg_kw = kruskal_score(avg_L)
    bad_kw = kruskal_score(bad_L)
    assert good_kw > avg_kw > bad_kw

    good_f = f_score(good_L)
    avg_f = f_score(avg_L)
    bad_f = f_score(bad_L)
    assert good_f > avg_f > bad_f

    good_mm = mood_median(good_L)
    avg_mm = mood_median(avg_L)
    bad_mm = mood_median(bad_L)
    assert good_mm > avg_mm > bad_mm