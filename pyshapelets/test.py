from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
from extractors.sax_shapelets import extract_shapelet as sax_extractor
import time
from visualization import visualize_shapelet
from collections import Counter

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])

adiac_data = None
for record in data:
	if record['name'] == 'Adiac':
		adiac_data = record['data']
data = adiac_data
print(data.shape)
#data = data.sample(25, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
print(Counter(y))
#X = X[X.columns[:25]]
start = time.time()
best_shapelet, best_dist, best_L, max_gain, max_gap = sax_extractor(X, y, sax_length=15, nr_candidates=10, max_len=50)
print('Took {} seconds'.format(time.time() - start)) 
print(best_shapelet, best_dist)
print(best_L)
visualize_shapelet(X.values, y.values, best_shapelet)
