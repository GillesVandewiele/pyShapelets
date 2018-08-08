import time
from collections import Counter, defaultdict
import warnings; warnings.filterwarnings('ignore')
import glob
import re
import ast

import numpy as np
import pandas as pd

from algorithms import ShapeletTransformer
from extractors.extractor import MultiGeneticExtractor
from data.load_all_datasets import load_data_train_test

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from tslearn.shapelets import ShapeletModel


def parse_shapelets(shapelets):
    shapelets = shapelets.replace(']', '],')[:-2]
    shapelets = re.sub(r'\s+', ', ', shapelets)
    shapelets = re.sub(r',+', ',', shapelets)
    shapelets = shapelets.replace('],[', '], [')
    shapelets = shapelets.replace('[,', '[')
    shapelets = '[' + shapelets + ']'
    shapelets = re.sub(r',\s+]', ']', shapelets)
    return ast.literal_eval(shapelets)

def fit_rf(X_distances_train, y_train, X_distances_test, y_test, out_path):
    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 25, 50, 100, 500], 'max_depth': [None, 3, 7, 15]})
    rf.fit(X_distances_train, y_train)
    
    hard_preds = rf.predict(X_distances_test)
    proba_preds = rf.predict_proba(X_distances_test)

    print("[RF] Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[RF] Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_rf_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_rf_proba.csv')

def fit_lr(X_distances_train, y_train, X_distances_test, y_test, out_path):                                                                                                                     
    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)
    
    hard_preds = lr.predict(X_distances_test)
    proba_preds = lr.predict_proba(X_distances_test)

    print("[LR] Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[LR] Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_lr_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_lr_proba.csv')

def fit_svm(X_distances_train, y_train, X_distances_test, y_test, out_path):
    svc = GridSearchCV(SVC(kernel='linear', probability=True), {'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    svc.fit(X_distances_train, y_train)
    
    hard_preds = svc.predict(X_distances_test)
    proba_preds = svc.predict_proba(X_distances_test)

    print("[SVM] Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[SVM] Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_svm_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_svm_proba.csv')

def fit_lts(X_train, y_train, X_test, y_test, shap_dict, reg, max_it, shap_out_path, pred_out_path, timing_out_path):
    # Fit LTS model, print metrics on test-set, write away predictions and shapelets
    clf = ShapeletModel(n_shapelets_per_size=shap_dict, 
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

    print('Learning shapelets took {}s'.format(learning_time))

    with open(shap_out_path, 'w+') as ofp:
        for shap in clf.shapelets_:
            ofp.write(str(np.reshape(shap, (-1))) + '\n')

    with open(timing_out_path, 'w+') as ofp:
        ofp.write(str(learning_time))

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    print('Max distance value = {}'.format(np.max(X_distances_train)))

    fit_rf(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    fit_svm(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

hyper_parameters_lts = {
    'Adiac':                     [0.3,  0.2,   3, 0.01, 10000],
    'Beef':                     [0.15, 0.125, 3, 0.01, 10000],
    'BeetleFly':                 [0.15, 0.125, 1, 0.01, 5000],
    'BirdChicken':                 [0.3,  0.075, 1, 0.1,  10000],
    'ChlorineConcentration':    [0.3,  0.2,   3, 0.01, 10000],
    'Coffee':                     [0.05, 0.075, 2, 0.01, 5000],
    'DiatomSizeReduction':         [0.3,  0.175, 2, 0.01, 10000],
    'ECGFiveDays':                 [0.05, 0.125, 2, 0.01, 10000],
    'FaceFour':                 [0.3,  0.175, 3, 1.0,  5000],
    'GunPoint':                 [0.15, 0.2,   3, 0.1,  10000],
    'ItalyPowerDemand':            [0.3,  0.2,   3, 0.01, 5000],
    'Lightning7':                 [0.05, 0.075, 3, 1,    5000],
    'MedicalImages':             [0.3,  0.2,   2, 1,    10000],
    'MoteStrain':                 [0.3,  0.2,   3, 1,    10000],
    #NOT AVAILABLE#'Otoliths':                 [0.15, 0.125, 3, 0.01, 2000],
    'SonyAIBORobotSurface1':     [0.3,  0.125, 2, 0.01, 10000],
    'SonyAIBORobotSurface2':     [0.3,  0.125, 2, 0.01, 10000],
    'Symbols':                     [0.05, 0.175, 1, 0.1,  5000],
    'SyntheticControl':         [0.15, 0.125, 3, 0.01, 5000],
    'Trace':                     [0.15, 0.125, 2, 0.1,  10000],
    'TwoLeadECG':                 [0.3,  0.075, 1, 0.1,  10000]
}

learning_sizes = defaultdict(list)
genetic_sizes = defaultdict(list)

metadata = sorted(load_data_train_test(), key=lambda x: x['train']['n_samples']**2*x['train']['n_features']**3)

for dataset in metadata:
    print(dataset['train']['name'], len(dataset['train']['name']), dataset['train']['name'] in hyper_parameters_lts)
    if dataset['train']['name'] not in hyper_parameters_lts: continue
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
    
    files = glob.glob('results/lts_vs_genetic/{}_genetic_shapelets*.txt'.format(dataset['train']['name']))
    if len(files):
        sizes = []
        for f in files:
            shaps = parse_shapelets(open(f, 'r').read())
            genetic_sizes[dataset['train']['name']].append(len(shaps))
            for s in shaps:
                sizes.append(len(s))
            
        shap_dict_cntr = Counter(np.random.choice(sizes, size=int(np.mean(genetic_sizes[dataset['train']['name']]))))
        shap_dict = {}
        for c in shap_dict_cntr:
            shap_dict[int(c)] = int(shap_dict_cntr[c])
            
        print(dataset['train']['name'], shap_dict)
            
        fit_lts(X_train, y_train, X_test, y_test, dict(shap_dict), reg, max_it,
                'results/lts_smaller/{}_learned_shapelets_{}.txt'.format(dataset['train']['name'], int(time.time())), 
                'results/lts_smaller/{}_learned_shapelets_predictions_{}.csv'.format(dataset['train']['name'], int(time.time())), 
                'results/lts_smaller/{}_learned_runtime_{}.csv'.format(dataset['train']['name'], int(time.time()))
        )

