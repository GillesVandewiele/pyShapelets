from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
from extractors.sax_shapelets import extract_shapelet as sax_extractor
from extractors.extractor import SAXExtractor, BruteForceExtractor, FastExtractor, LearningExtractor, GeneticExtractor, ParticleSwarmExtractor
import time
from visualization import visualize_shapelet
from collections import Counter
import numpy as np
import util

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])

adiac_data = None
for record in data:
	if record['name'] == 'ArrowHead':
		adiac_data = record['data']
		break
data = adiac_data

data = data.sample(75, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
map_dict = {}
for i, c in enumerate(np.unique(y)):
	map_dict[c] = i
y = y.map(map_dict)
X = X[X.columns[50:150]]


"""
start = time.time()
extractor = SAXExtractor()
best_shapelets = extractor.extract(X, y, min_len=25, max_len=50, metric='f')
L = []
for ts, label in zip(X.values, y):
	L.append((util.sdist(best_shapelets[0], ts), label))
L = sorted(L, key=lambda x: x[0])
print(util.calculate_ig(L))
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0])
"""

"""
start = time.time()
extractor = ParticleSwarmExtractor(particles=25)
best_shapelets = extractor.extract(X, y, min_len=25, max_len=50, metric='f', nr_shapelets=1)
print(best_shapelets)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0])

start = time.time()
extractor = GeneticExtractor(population_size=25)
best_shapelets = extractor.extract(X, y, min_len=25, max_len=50, metric='f', nr_shapelets=1)
print(best_shapelets)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0])
"""


from algorithms import ShapeletTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

skf = StratifiedKFold(n_splits=3)
for train_idx, test_idx in skf.split(X, y):
	X_train = X.iloc[train_idx, :]
	y_train = y.iloc[train_idx]
	X_test = X.iloc[test_idx, :]
	y_test = y.iloc[test_idx]

	shap_transformer = ShapeletTransformer(nr_shapelets=10, method='sax', min_len=20, max_len=40)
	shap_transformer.fit(X_train, y_train)
	X_distances_train = shap_transformer.transform(X_train)
	X_distances_test = shap_transformer.transform(X_test)

	rf = RandomForestClassifier()
	rf.fit(X_distances_train, y_train)

	print(confusion_matrix(y_test, rf.predict(X_distances_test)))

	shap_transformer = ShapeletTransformer(nr_shapelets=10, method='genetic', min_len=20, max_len=40)
	shap_transformer.fit(X_train, y_train)
	X_distances_train = shap_transformer.transform(X_train)
	X_distances_test = shap_transformer.transform(X_test)

	rf = RandomForestClassifier()
	rf.fit(X_distances_train, y_train)

	print(confusion_matrix(y_test, rf.predict(X_distances_test)))



"""
start = time.time()
extractor = SAXExtractor(sax_length=8)
best_shapelets = extractor.extract(X, y, min_len=8, metric='f')
print(best_shapelets)
print('Took {} seconds'.format(time.time() - start)) 
#visualize_shapelet(X.values, y, best_shapelets[0])

start = time.time()
extractor = SAXExtractor(sax_length=8)
best_shapelets = extractor.extract(X, y, min_len=8, metric='ig')
print(best_shapelets)
print('Took {} seconds'.format(time.time() - start)) 

start = time.time()
extractor = SAXExtractor(sax_length=8)
best_shapelets = extractor.extract(X, y, min_len=8, metric='mm')
print(best_shapelets)
print('Took {} seconds'.format(time.time() - start)) 
"""

"""
start = time.time()
extractor = LearningExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
#visualize_shapelet(X.values, y, best_shapelets[0])

start = time.time()
extractor = FastExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
#visualize_shapelet(X.values, y, best_shapelets[0])

start = time.time()
extractor = BruteForceExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
#visualize_shapelet(X.values, y, best_shapelets[0])
"""