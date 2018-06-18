import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import ShapeletTransformer
from extractors.extractor import MultiGeneticExtractor
from data.load_all_datasets import load_data_train_test

from sklearn.metrics import accuracy_score
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

# For each dataset we specify the:
#    * Number of shapelets to extract of each length (specified as a fraction of TS length)
#    * Minimal shapelet length (specified as a fraction of TS length)
#    * Different scales of shapelet lengths
#    * Weight regularizer
#    * Number of iterations
hyper_parameters_lts = {
	'Adiac': 					[0.3,  0.2,   3, 0.01, 10000],
	'Beef': 					[0.15, 0.125, 3, 0.01, 10000],
	'BeetleFly': 				[0.15, 0.125, 1, 0.01, 5000],
	'BirdChicken': 				[0.3,  0.075, 1, 0.1,  10000],
	'Chlorine': 				[0.3,  0.2,   3, 0.01, 10000],
	'Coffee': 					[0.05, 0.075, 2, 0.01, 5000],
	'DiatomSizeReduction': 		[0.3,  0.175, 2, 0.01, 10000],
	'ECGFiveDays': 				[0.05, 0.125, 2, 0.01, 5000],
	'FaceFour': 				[0.3,  0.175, 3, 1.0,  5000],
	'GunPoint': 				[0.15, 0.2,   3, 0.1,  10000],
	'ItalyPowerDemand':			[0.3,  0.2,   3, 0.01, 5000],
	'Lightning7': 				[0.05, 0.075, 3, 1,    5000],
	'MedicalImages': 			[0.3,  0.2,   2, 1,    10000],
	'MoteStrain': 				[0.3,  0.2,   3, 1,    10000],
	'Otoliths': 				[0.15, 0.125, 3, 0.01, 2000],
	'SonyAIBORobotSurface1': 	[0.3,  0.125, 2, 0.01, 10000],
	'SonyAIBORobotSurface2': 	[0.3,  0.125, 2, 0.01, 10000],
	'Symbols': 					[0.05, 0.175, 1, 0.1,  5000],
	'SyntheticControl': 		[0.15, 0.125, 3, 0.01, 5000],
	'Trace': 					[0.15, 0.125, 2, 0.1,  10000],
	'TwoLeadECG': 				[0.3,  0.075, 1, 0.1,  10000]
}

metadata = sorted(load_data_train_test(), key=lambda x: x['train']['n_samples']**2*x['train']['n_features']**3)
result_vectors = []

for dataset in metadata:
    if dataset['train']['name'] not in hyper_parameters_lts: continue

    print(dataset['train']['name'])

    # Load the training and testing dataset (features + label vector)
    train_df = pd.read_csv(dataset['train']['data_path'])
    test_df = pd.read_csv(dataset['test']['data_path'])
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target']

    map_dict = {}
    for j, c in enumerate(np.unique(y_train)):
        map_dict[c] = j
    y_train = y_train.map(map_dict) 
    y_test = y_test.map(map_dict)

    y_train = y_train.values
    y_test = y_test.values

    nr_shap, l, r, reg, max_it = hyper_parameters_lts[dataset['train']['name']]
    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), l, r
    )
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=1,
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

    learning_accuracy = accuracy_score(y_test, clf.predict(X_test))

    with open('results/lts_vs_genetic/{}_learned_shapelets.txt'.format(dataset['train']['name']), 'w+') as ofp:
    	for shap in clf.shapelets_:
    		ofp.write(str(shap) + '\n')

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500]})
    rf.fit(X_distances_train, y_train)

    learning_accuracy_rf = accuracy_score(y_test, rf.predict(X_distances_test))

    svc = GridSearchCV(SVC(kernel='linear'), {'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    svc.fit(X_distances_train, y_train)

    learning_accuracy_svm = accuracy_score(y_test, svc.predict(X_distances_test))

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    learning_accuracy_lr = accuracy_score(y_test, lr.predict(X_distances_test))

    print(learning_accuracy, learning_accuracy_lr, learning_accuracy_svm, learning_accuracy_rf)

    print('Learning shapelets took {}s'.format(learning_time))
    print('Accuracies are equal to: LR={}, RF={}, SVM={}'.format(learning_accuracy_lr, learning_accuracy_rf, learning_accuracy_svm))

    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=125, iterations=500, wait=25, plot=False)
    shap_transformer = ShapeletTransformer(method=genetic_extractor, min_len=4, max_len=X_train.shape[1], nr_shapelets=X_train.shape[1]//2, metric='ig')
    start = time.time()
    shap_transformer.fit(X_train, y_train)
    genetic_time = time.time() - start


    with open('results/lts_vs_genetic/{}_genetic_shapelets.txt'.format(dataset['train']['name']), 'w+') as ofp:
    	for shap in shap_transformer.shapelets:
    		ofp.write(str(shap) + '\n')

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500]})
    rf.fit(X_distances_train, y_train)

    genetic_accuracy_rf = accuracy_score(y_test, rf.predict(X_distances_test))

    svc = GridSearchCV(SVC(kernel='linear'), {'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    svc.fit(X_distances_train, y_train)

    genetic_accuracy_svm = accuracy_score(y_test, svc.predict(X_distances_test))

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    genetic_accuracy_lr = accuracy_score(y_test, lr.predict(X_distances_test))

    print('RF Acc:', accuracy_score(y_test, rf.predict(X_distances_test)))
    print('SVM Acc:', accuracy_score(y_test, svc.predict(X_distances_test)))

    result_vectors.append([dataset['train']['name']] + [genetic_time, genetic_accuracy_lr, genetic_accuracy_rf, 
                                                        genetic_accuracy_svm, learning_time, 
                                                        learning_accuracy_lr, learning_accuracy_rf, 
                                                        learning_accuracy_svm])

    print('Genetic extraction of shapelets took {}s'.format(genetic_time))
    print('Accuracies are equal to: LR={}, RF={}, SVM={}'.format(genetic_accuracy_lr, genetic_accuracy_rf, genetic_accuracy_svm))

results_df = pd.DataFrame(result_vectors)
results_df.columns = ['Dataset Name', 'Genetic Time', 'Genetic Accuracy (LR)', 'Genetic Accuracy (RF)', 
                      'Genetic Accuracy (SVM)', 'Learning Time', 'Learning Accuracy (LR)',
                      'Learning Accuracy (RF)', 'Learning Accuracy (SVM)']
results_df.to_csv('results/lts_vs_genetic/results.csv')