import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import ShapeletTransformer
from extractors.extractor import MultiGeneticExtractor
from data.load_all_datasets import load_data_train_test

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from tslearn.shapelets import ShapeletModel


def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_shapelets, l, r):
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        d[shp_sz] = n_shapelets
    return d


np.random.seed(1337)
random.seed(1337)

TRAIN_PATH = 'data/partitioned/Coffee/Coffee_train.csv'
TEST_PATH = 'data/partitioned/Coffee/Coffee_test.csv'

# Load the training and testing dataset (features + label vector)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target']
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target']

map_dict = {}
for j, c in enumerate(np.unique(y_train)):
    map_dict[c] = j
y_train = y_train.map(map_dict) 
y_test = y_test.map(map_dict)

print(set(y_train), set(y_test))

y_train = y_train.values
y_test = y_test.values

nr_shap, l, r, reg, max_it = 0.05, 0.075, 2, 0.01, 5000

measurements = []

for _nr_shap in np.arange(0.05, 0.55, 0.05):

    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(_nr_shap*X_train.shape[1]), l, r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=8,
                        optimizer='sgd', weight_regularizer=reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[K]', _nr_shap, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([_nr_shap, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of LTS on hyper-parameter K')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['nr_shap', 'accuracy'])
measurements_df.to_csv('lts_param_K.csv')

measurements = []

for _l in np.arange(0.075, 0.4, 0.075):

    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), _l, r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=8,
                        optimizer='sgd', weight_regularizer=reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[L', _l, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([_l, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of LTS on hyper-parameter L (ItalyPowerDemand)')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['l', 'accuracy'])
measurements_df.to_csv('lts_param_L.csv')


measurements = []

for _r in range(1, 6):

    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), l, _r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=8,
                        optimizer='sgd', weight_regularizer=reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[R]', _r, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([_r, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of LTS on hyper-parameter R (ItalyPowerDemand)')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['r', 'accuracy'])
measurements_df.to_csv('lts_param_R.csv')


measurements = []

for _reg in np.arange(0., 0.55, 0.05):

    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), l, r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=8,
                        optimizer='sgd', weight_regularizer=_reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[REG]', _reg, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([_reg, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of LTS on hyper-parameter lambda (ItalyPowerDemand)')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['reg', 'accuracy'])
measurements_df.to_csv('lts_param_lambda.csv')


measurements = []

for _max_it in range(1000, 11000, 1000):

    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), l, r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=_max_it, verbose_level=0, batch_size=8,
                        optimizer='sgd', weight_regularizer=reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[Max Iter]', _max_it, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([_max_it, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of LTS on hyper-parameter max. iter. (ItalyPowerDemand)')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['maxiter', 'accuracy'])
measurements_df.to_csv('lts_param_maxiter.csv')