from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
from extractors.sax_shapelets import extract_shapelet as sax_extractor
from extractors.extractor import SAXExtractor, BruteForceExtractor, FastExtractor, LearningExtractor
import time
from visualization import visualize_shapelet
from collections import Counter
import numpy as np

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])

adiac_data = None
for record in data:
	if record['name'] == 'GunPoint':
		adiac_data = record['data']
data = adiac_data


data = data.sample(25, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
map_dict = {}
for i, c in enumerate(np.unique(y)):
	map_dict[c] = i
y = y.map(map_dict)
X = X[X.columns[:25]]


start = time.time()
extractor = SAXExtractor(sax_length=8)
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0][0])

start = time.time()
extractor = LearningExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0][0])

start = time.time()
extractor = FastExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0][0])

start = time.time()
extractor = BruteForceExtractor()
best_shapelets = extractor.extract(X, y, min_len=8)
print('Took {} seconds'.format(time.time() - start)) 
visualize_shapelet(X.values, y, best_shapelets[0][0])
