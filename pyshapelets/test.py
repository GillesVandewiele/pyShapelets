from data.load_all_datasets import load_data
from extractors.brute_force import extract_shapelet
import time

data = sorted(load_data(), key=lambda x: x['n_samples']*x['n_features'])[0]['data']
data = data.sample(25)
X = data.drop('target', axis=1)
y = data['target']
X = X[X.columns[:25]]
start = time.time()
extract_shapelet(X, y, min_len=10, max_len=11)
print('Took {} seconds'.format(time.time() - start))  #Took 134.85634970664978 seconds