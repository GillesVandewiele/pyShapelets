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

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])

#for i in range(len(data)):
#    if data[i]['name'] == 'Beef':
#        data = [data[i]]
#        break

extractors = [
    #LearningExtractor(),
    #ParticleSwarmExtractor2(particles=50, iterations=100, wait=5),
    #ParticleSwarmExtractor(particles=50, iterations=100, wait=5),
    #SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
    #             iterations=5, mask_size=3),
    MultiGeneticExtractor(population_size=5, iterations=100, verbose=True,
                     mutation_prob=0.25, crossover_prob=0.4, wait=5),
    #FastExtractor()

]

for i in range(10):
    X = data[i]['data'].drop('target', axis=1)
    y = data[i]['data'].loc[X.index, 'target']
    map_dict = {}
    for j, c in enumerate(np.unique(y)):
        map_dict[c] = j
    y = y.map(map_dict) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)

    """
    print('Fitting tree on {} ({})'.format(data[i]['name'], X_train.shape))
    for extractor in extractors:
        print('\t Using the {} extractor'.format(extractor))
        shapelet_tree = ShapeletTreeClassifier(method=extractor, max_len=data[i]['n_features']//2, metric='ig')
        start = time.time()
        shapelet_tree.fit(X_train, y_train)
        print(confusion_matrix(y_test, shapelet_tree.predict(X_test.values)))
        print('\t Took {} seconds'.format(time.time() - start)) 
    """

    print('Fitting shapelet transform on {} ({})'.format(data[i]['name'], X_train.shape))

    clf = ShapeletModel(n_shapelets_per_size=grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0], ts_sz=X_train.shape[1], n_classes=3, l=0.1, r=2), 
                        max_iter=100, verbose_level=1,
                        optimizer='adagrad',
                        weight_regularizer=0.0)
    clf.fit(
        np.reshape(
            X_train.values, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    predictions = clf.predict(np.reshape(
            X_test.values, 
            (X_test.shape[0], X_test.shape[1], 1)
    ))
    print(confusion_matrix(y_test, predictions))


    for extractor in extractors:
        print('\t Using the {} extractor'.format(extractor))
        shap_transformer = ShapeletTransformer(method=extractor, max_len=data[i]['n_features']//2, nr_shapelets=5, metric='ig')
        start = time.time()
        shap_transformer.fit(X_train, y_train)
        print(shap_transformer.shapelets)
        X_distances_train = shap_transformer.transform(X_train)
        X_distances_test = shap_transformer.transform(X_test)

        rf = RandomForestClassifier()
        rf.fit(X_distances_train, y_train)

        print(confusion_matrix(y_test, rf.predict(X_distances_test)))
        print('\t Took {} seconds'.format(time.time() - start)) 

