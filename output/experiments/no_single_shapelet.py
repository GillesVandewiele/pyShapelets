import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
	+ os.sep 
	+ 'pyshapelets'
)
from extractors.extractor import BruteForceExtractor, GeneticExtractor, ParticleSwarmExtractor
import util

# Define four time series
ts1 = [0]*2 + [-1]*8 + [0]*5 + [1]*8 + [0]*2
ts2 = [0]*2 + [1]*8 + [0]*5 + [-1]*8 + [0]*2

ts3 = [0]*2 + [-1]*8 + [0]*5 + [-1]*8 + [0]*2
ts4 = [0]*2 + [1]*8 + [0]*5 + [1]*8 + [0]*2

X = np.array([ts1, ts2, ts3, ts4], dtype='float')
y = [0, 0, 1, 1]
# we cannot find a shapelet of max length 10 that gives
# perfect separation in the data. 


extractor = BruteForceExtractor()
bad_shapelet = extractor.extract(X, y, metric='ig')[0]
L = []
for k in range(len(X)):
	D = X[k, :]
	dist = util.sdist(bad_shapelet, D)
	L.append((dist, y[k]))
print(L)

"""
from data.load_all_datasets import load_data
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
"""


extractor = GeneticExtractor( population_size=100, iterations=500, wait=5 )
gen_shapelet = extractor.extract(X, y, metric='ig')[0]
L = []
for k in range(len(X)):
	D = X[k, :]
	dist = util.sdist(gen_shapelet, D)
	L.append((dist, y[k]))
print(L)

# This shapelet, which is not directly in the data
# can offer a perfect separation
good_shapelet = [1]*1 + [0]*5 + [1]*1
#good_shapelet = util.z_norm(good_shapelet)
L = []
for k in range(len(X)):
	D = X[k, :]
	dist = util.sdist(good_shapelet, D)
	L.append((dist, y[k]))
print(L)

f, axarr = plt.subplots(7, sharex=True, sharey=True)
x = range(len(ts1))
axarr[0].plot(x, ts1, color='r')
axarr[1].plot(x, ts2, color='r')
axarr[2].plot(x, ts3)
axarr[3].plot(x, ts4)
axarr[4].plot(range(len(good_shapelet)), good_shapelet)
axarr[5].plot(range(len(bad_shapelet)), bad_shapelet)
axarr[6].plot(range(len(gen_shapelet)), gen_shapelet)
plt.show()