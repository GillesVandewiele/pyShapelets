from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
from extractors.sax_shapelets import extract_shapelet as sax_extractor
from extractors.extractor import SAXExtractor, BruteForceExtractor, FastExtractor, LearningExtractor, GeneticExtractor, ParticleSwarmExtractor, ParticleSwarmExtractor2, MultiGeneticExtractor
from algorithms import ShapeletTreeClassifier, ShapeletTransformer
import time
from visualization import visualize_shapelet
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import util
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
import warnings; warnings.filterwarnings('ignore')


def estimate_min_max(X, y, extractor, min_perc=10, max_perc=90):
    shapelet_lengths = []
    for _ in range(10):
        rand_idx = np.random.choice(range(len(X)), size=10, replace=False)
        X_sub = X[rand_idx, :]
        y_sub = y[rand_idx]
        shapelet_lengths += [len(x) for x in extractor.extract(X_sub, y_sub, min_len=3, nr_shapelets=10)]
        
    _min = int(np.percentile(shapelet_lengths, min_perc))
    _max = int(np.percentile(shapelet_lengths, max_perc))
    if _min == _max:
        _min -= 1
    return _min, _max


data = sorted(load_data(), key=lambda x: x['n_samples']**2*x['n_features']**3)

for i in range(10):
    X = data[i]['data'].drop('target', axis=1)
    y = data[i]['data'].loc[X.index, 'target']
    map_dict = {}
    for j, c in enumerate(np.unique(y)):
        map_dict[c] = j
    y = y.map(map_dict) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    print('Fitting shapelet transform on {} ({})'.format(data[i]['name'], X_train.shape))

    extractor = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
                             iterations=5, mask_size=3)

    _min, _max = estimate_min_max(X_train, y_train, extractor)
    print(_min, _max)

    print('\t Using the applying transformation on top-k shapelets'.format(extractor))
    shap_transformer = ShapeletTransformer(method=extractor, min_len=_min, max_len=_max, nr_shapelets=data[i]['n_features']//2, metric='f')
    start = time.time()
    shap_transformer.fit(X_train, y_train)
    print(shap_transformer.shapelets)
    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = RandomForestClassifier()
    rf.fit(X_distances_train, y_train)

    print(confusion_matrix(y_test, rf.predict(X_distances_test)))
    print('\t Took {} seconds'.format(time.time() - start)) 

    input()

