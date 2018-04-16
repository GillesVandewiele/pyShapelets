from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet as bf_extractor
from extractors.fast_shapelets import extract_shapelet as fast_extractor
import time
from visualization import visualize_shapelet

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])[0]['data']
data = data.sample(50, random_state=1337)
X = data.drop('target', axis=1)
y = data['target']
X = X[X.columns[:50]]
start = time.time()
# Took 37.8997585773468 seconds (25x25 matrix, w/o pruning)
# Took 149.82520270347595 seconds (50x25 matrix, w/o pruning)
# 50x50 w/o pruning: Took 1117.8448395729065 seconds  !! (with pruning: 1664.9176104068756 seconds)
best_shapelet, best_dist, best_L, max_gain, max_gap = fast_extractor(X, y, enable_pruning=True)
print('Took {} seconds'.format(time.time() - start)) 
print(best_shapelet, best_dist)
print(best_L)
visualize_shapelet(X.values, y.values, best_shapelet)
#start = time.time()
#best_shapelet, best_dist, best_L, max_gain, max_gap = bf_extractor(X, y)
#print('Took {} seconds'.format(time.time() - start))  #Took 134.85634970664978 seconds
#print(best_shapelet, best_dist)
#print(best_L)