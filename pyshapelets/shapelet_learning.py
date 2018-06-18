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

from keras.utils import to_categorical

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])[9]

X = data['data'].drop('target', axis=1)
y = data['data'].loc[X.index, 'target']
map_dict = {}
for j, c in enumerate(np.unique(y)):
    map_dict[c] = j
y = y.map(map_dict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337, stratify=y)

print(X_train.shape)

print(Counter(y_train))

print(grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0], ts_sz=X_train.shape[1], n_classes=len(set(y)), l=0.05, r=10))

clf = ShapeletModel(n_shapelets_per_size=grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0], ts_sz=X_train.shape[1], n_classes=len(set(y)), l=0.1, r=10), 
                    max_iter=1000, verbose_level=1,
                    optimizer='adam',
                    weight_regularizer=0.0)
clf.fit(
    np.reshape(
        X_train.values, 
        (X_train.shape[0], X_train.shape[1], 1)
    ), 
    to_categorical(y_train)
)
predictions = np.argmax(clf.predict(np.reshape(
        X_test.values, 
        (X_test.shape[0], X_test.shape[1], 1)
)), axis=1)
print(confusion_matrix(y_test, predictions))