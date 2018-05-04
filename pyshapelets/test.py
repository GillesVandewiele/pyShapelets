from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
from extractors.sax_shapelets import extract_shapelet as sax_extractor
from extractors.extractor import SAXExtractor
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
print(data.shape)
data = data.sample(25, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
map_dict = {}
for i, c in enumerate(np.unique(y)):
	map_dict[c] = i
y = y.map(map_dict)
print(Counter(y))
#X = X[X.columns[:25]]
start = time.time()
extractor = SAXExtractor()
best_shapelet, best_dist, best_L, max_gain, max_gap = extractor.extract(X, y)
#best_shapelet, best_dist, best_L, max_gain, max_gap = sax_extractor(X, y, sax_length=15, nr_candidates=10, max_len=30)
print('Took {} seconds'.format(time.time() - start)) 
print(best_shapelet, best_dist)
print(best_L)
visualize_shapelet(X.values, y.values, best_shapelet)
