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


data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])[0]

X = data['data'].drop('target', axis=1)
y = data['data'].loc[X.index, 'target']
map_dict = {}
for j, c in enumerate(np.unique(y)):
    map_dict[c] = j
y = y.map(map_dict)

extractor = MultiGeneticExtractor(population_size=5, iterations=100, verbose=True,
                                  mutation_prob=0.25, crossover_prob=0.4, wait=5)

shap_transformer = ShapeletTransformer(method=extractor, max_len=data['n_features']//2, nr_shapelets=5, metric='ig')
shap_transformer.fit(X, y)
X_distances = shap_transformer.transform(X)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

x1 = X_distances[:, 0]
x2 = X_distances[:, 1]
plt.hist2d(x1, x2)
plt.show()