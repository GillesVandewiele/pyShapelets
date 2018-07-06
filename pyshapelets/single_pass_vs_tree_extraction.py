import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.load_all_datasets import load_data_train_test
from algorithms import ShapeletTreeClassifier, ShapeletTransformer
import util
from extractors.extractor import SAXExtractor

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def estimate_min_max(X, y, extractor, min_perc=25, max_perc=75, min_len=3, 
                     max_len=None, iterations=10):
    shapelet_lengths = []
    for _ in range(iterations):
        rand_idx = np.random.choice(range(len(X)), size=10, replace=False)
        X_sub = X[rand_idx, :]
        y_sub = y[rand_idx]
        shapelet_lengths += [len(x) for x in extractor.extract(X_sub, y_sub, 
                                                               min_len=min_len, 
                                                               max_len=max_len, 
                                                               nr_shapelets=10)]
        
    _min = int(np.percentile(shapelet_lengths, min_perc))
    _max = max(_min + 1, int(np.percentile(shapelet_lengths, max_perc)))
    return _min, _max

def extract_shapelets_with_tree(X_train, y_train, extractor, min_len, max_len):
    shap_tree_clf = ShapeletTreeClassifier(method=extractor, min_len=min_len,
                                           max_len=max_len)
    shap_tree_clf.fit(X_train, y_train)
    return shap_tree_clf.tree.extract_all_shapelets()



metadata = sorted(load_data_train_test(), key=lambda x: x['train']['n_samples']**2*x['train']['n_features']**3)
for dataset in metadata:
    if dataset['train']['name'] not in ['DiatomSizeReduction', 'ArrowHead', 'Beef', 'MoteStrain', 'Coffee', 'GunPoint']: continue
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

    # Create the extractor & determine min and max shapelet length heuristically
    extractor = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
                              iterations=5, mask_size=3)
    _min, _max = estimate_min_max(X_train, y_train, extractor)

    # Extract shapelets using Shapelet Tree and fit RF
    tree_shapelets = extract_shapelets_with_tree(X_train, y_train, extractor,
                                                 _min, _max)
    shap_transformer = ShapeletTransformer(method=extractor, min_len=_min, 
                                           max_len=_max)
    shap_transformer.shapelets = [np.array(x) for x in tree_shapelets]

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(
    	RandomForestClassifier(random_state=1337),
    	{'n_estimators': [5, 10, 50, 100, 250, 500]}
    )
    rf.fit(X_distances_train, y_train)

    # Write away the predictions + a plot from all shapelets
    rf_preds = pd.DataFrame(rf.predict_proba(X_distances_test))
    rf_preds.to_csv('results/transform_vs_tree/{}_rf_preds_tree.csv'.format(dataset['train']['name']))
    with open('results/transform_vs_tree/{}_shaps_tree.txt'.format(dataset['train']['name']), 'w') as ofp:
        for shapelet in tree_shapelets:
            ofp.write(str(shapelet)+'\n')

    print('Tree extraction:')
    print(confusion_matrix(y_test, rf.predict(X_distances_test)))


    # Do the same, but extract the shapelets in a single pass (features//2 shapelets)
    shap_transformer = ShapeletTransformer(method=extractor, min_len=_min, 
                                           max_len=_max, nr_shapelets=dataset['train']['n_features']//2)
    shap_transformer.fit(X_train, y_train)
    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(
    	RandomForestClassifier(random_state=1337),
    	{'n_estimators': [5, 10, 50, 100, 250, 500]}
    )
    rf.fit(X_distances_train, y_train)

    rf_preds = pd.DataFrame(rf.predict_proba(X_distances_test))
    rf_preds.to_csv('results/transform_vs_tree/{}_rf_preds_transform.csv'.format(dataset['train']['name']))
    with open('results/transform_vs_tree/{}_shaps_transform.txt'.format(dataset['train']['name']), 'w') as ofp:
        for shapelet in shap_transformer.shapelets:
            ofp.write(str(shapelet)+'\n')


    print('Single pass:')
    print(confusion_matrix(y_test, rf.predict(X_distances_test)))


