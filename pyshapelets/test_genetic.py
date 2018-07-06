import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import ShapeletTransformer
from extractors.extractor import GeneticExtractor, MultiGeneticExtractor, SAXExtractor, LearningExtractor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tslearn.shapelets import ShapeletModel
from mstamp.mstamp_stomp import mstamp as mstamp_stomp
from mstamp.mstamp_stamp import mstamp as mstamp_stamp

# Prepare all the datasets
datasets = []
meta_info = [
    #('ItalyPowerDemand', None, None, None, 13, 18),    	# (Acc: 0.93, Time: 35s)
    #('SonyAIBORobotSurface1', None, None, None, 17, 54),   # (Acc: 0.8519, Time: 218s)
    #('SonyAIBORobotSurface2', None, None, None, 32, 63),   # (Acc: 0.8519, Time: 131s)
    #('MoteStrain', None, None, None, 16, 33),    			# (Acc: 0.8458, Time: xx)
    ('Beef', None, None, None, 15, 128),    				# (Acc: 0.90, Time: xx)
]

"""
Short runs:
------------
                       0           1         2
0       ItalyPowerDemand   36.171110  0.896016
1  SonyAIBORobotSurface1   26.520345  0.883527
2  SonyAIBORobotSurface2   33.203732  0.838405
3             MoteStrain   43.249932  0.802716
4                   Beef  111.535088  0.533333

Longer runs:
------------
0       ItalyPowerDemand  186.922645  0.952381
1  SonyAIBORobotSurface1  132.779138  0.921797
2  SonyAIBORobotSurface2  164.222865  0.803778
3             MoteStrain  228.977391  0.753195
4                   Beef  351.845052  0.500000

"""

"""
ItalyPowerDemand
SonyAIBORobotSurface1
SonyAIBORobotSurface2
MoteStrain
TwoLeadECG
ECGFiveDays
CBF
GunPoint
ECG200
DiatomSizeReduction
"""

def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_shapelets, l, r):
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        d[shp_sz] = n_shapelets
    return d

def estimate_min_max(X, y, extractor, min_perc=25, max_perc=75, min_len=3, max_len=None, iterations=5):
    shapelet_lengths = []
    for _ in range(iterations):
        rand_idx = np.random.choice(range(len(X)), size=10, replace=False)
        X_sub = X[rand_idx, :]
        y_sub = y[rand_idx]

        map_dict = {}
        for j, c in enumerate(np.unique(y_sub)):
            map_dict[c] = j
        y_sub = np.vectorize(map_dict.get)(y_sub)

        shapelet_lengths += [len(x) for x in extractor.extract(X_sub, y_sub, min_len=min_len, max_len=max_len, nr_shapelets=10)]
        
    _min = int(np.percentile(shapelet_lengths, min_perc))
    _max = int(np.percentile(shapelet_lengths, max_perc))
    if _min == _max:
        _max += 1
    print('Estimated a minimum and maximum length:', _min, _max)
    return _min, _max

result_vectors = []
for dataset_name, start_idx, end_idx, samples_per_class, min_len, max_len in meta_info:
    print(dataset_name)
    train_path = '/home/giles/Projects/pyShapelets/pyshapelets/data/partitioned/{}/{}_train.csv'.format(dataset_name, dataset_name)
    test_path = '/home/giles/Projects/pyShapelets/pyshapelets/data/partitioned/{}/{}_test.csv'.format(dataset_name, dataset_name)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(train_df.shape)
    print(Counter(train_df['target']))

    indices = list(train_df.index)
    if samples_per_class is not None:
        indices = []
        for c in set(train_df['target']):
            indices += list(np.random.choice(train_df[train_df['target'] == c].index, size=samples_per_class, replace=False))

    train_df = train_df.loc[indices, :]

    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target']

    """
    for size in (5, X_train.shape[1]//2, 10):
        ts = X_train.flatten()
        start = time.time()
        matrix_profile, _ = mstamp_stomp(ts, size)
        motif_idx = matrix_profile[0, :].argsort()[-1]
        print(ts[motif_idx:motif_idx + size])
        print(time.time() - start)
        start = time.time()
        matrix_profile = mstamp_stamp(ts, size)
        motif_idx = matrix_profile[0, :].argsort()[-1]
        print(ts[motif_idx:motif_idx + size])
        print(time.time() - start)
    """


    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = X_train.shape[1]

    X_train = X_train[:, start_idx:end_idx]
    X_test = X_test[:, start_idx:end_idx]

    map_dict = {}
    for j, c in enumerate(np.unique(y_train)):
        map_dict[c] = j
    y_train = y_train.map(map_dict) 
    y_test = y_test.map(map_dict)

    y_train = y_train.values
    y_test = y_test.values

    sax_extractor = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
                                  iterations=5, mask_size=3)

    if min_len is None or max_len is None:
        _min, _max = estimate_min_max(X_train, y_train, sax_extractor)
    else:
        _min, _max = min_len, max_len

    """
    shap_transformer = ShapeletTransformer(method=sax_extractor, min_len=_min, max_len=_max, nr_shapelets=5, metric='ig')
    start = time.time()
    shap_transformer.fit(X_train, y_train)
    sax_time = time.time() - start

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = RandomForestClassifier()
    rf.fit(X_distances_train, y_train)

    sax_accuracy = accuracy_score(y_test, rf.predict(X_distances_test))
	"""

    """
    shapelet_dict = grabocka_params_to_shapelet_size_dict(
                X_train.shape[0], X_train.shape[1], int(0.15*X_train.shape[1]), 0.125, 3
    )
    print(shapelet_dict)
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=10000, verbose_level=1, batch_size=1,
                        optimizer='sgd', weight_regularizer=0.01)
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )

    print('Learning shapelets on dataset {} || Accuracy = {}'.format(
        dataset_name, accuracy_score(y_test, clf.predict(X_test))
    ))

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500]})
    rf.fit(X_distances_train, y_train)

    print(accuracy_score(y_test, rf.predict(X_distances_test)))
    input()
    """

    """
    for l in np.arange(0.05, 1.05, 0.05):
        for r in range(1, 11):
            try:
                shapelet_dict = grabocka_params_to_shapelet_size_dict(
                    X_train.shape[0], X_train.shape[1], len(set(y_train)), l, r
                )
                clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                                    max_iter=1000, verbose_level=0,
                                    optimizer='sgd')
                extracted_shapelets = clf.fit(
                    np.reshape(
                        X_train, 
                        (X_train.shape[0], X_train.shape[1], 1)
                    ), 
                    y_train
                ).shapelets_

                shap_transformer = ShapeletTransformer()
                shap_transformer.shapelets = extracted_shapelets

                X_distances_train = shap_transformer.transform(X_train)
                X_distances_test = shap_transformer.transform(X_test)

                rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500]})
                rf.fit(X_distances_train, y_train)

                print('Learning shapelets on dataset {} with l={}, r={} || Accuracy = {}'.format(
                    dataset_name, l, r, accuracy_score(y_test, rf.predict(X_distances_test))
                ))
            except:
                print(l, r, 'failed')
    """


    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=20, iterations=500, wait=25, plot=True)
    shap_transformer = ShapeletTransformer(method=genetic_extractor, min_len=_min, max_len=_max, nr_shapelets=X_train.shape[1]//2, metric='ig')
    start = time.time()
    shap_transformer.fit(X_train, y_train)
    genetic_time = time.time() - start

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500]})
    rf.fit(X_distances_train, y_train)

    genetic_accuracy = accuracy_score(y_test, rf.predict(X_distances_test))
    result_vectors.append([dataset_name] + [genetic_time, genetic_accuracy])

    svc = SVC(kernel='linear')
    svc.fit(X_distances_train, y_train)

    print('RF Acc:', accuracy_score(y_test, rf.predict(X_distances_test)))
    print('SVM Acc:', accuracy_score(y_test, svc.predict(X_distances_test)))

results_df = pd.DataFrame(result_vectors)
print(results_df)
#results_df.columns = ['Dataset Name', 'SAX Time', 'SAX Accuracy', 'Genetic Time', 'Genetic Accuracy']
results_df.to_csv('results.csv')