from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
import time

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])[0]['data']
data = data.sample(25, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
X = X[X.columns[:50]]
start = time.time()
best_shapelet, best_dist, best_L, max_gain, max_gap = fast_extractor(X, y, min_len=10, max_len=11)
print('Took {} seconds'.format(time.time() - start)) 
print(best_shapelet, best_dist)
print(best_L)
start = time.time()
best_shapelet, best_dist, best_L, max_gain, max_gap = bf_extractor(X, y, min_len=10, max_len=11)
print('Took {} seconds'.format(time.time() - start))  #Took 134.85634970664978 seconds
print(best_shapelet, best_dist)
print(best_L)