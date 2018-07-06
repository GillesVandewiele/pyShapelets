import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import tqdm


def extract_shapelet(timeseries, labels, min_len=None, max_len=None):
	if min_len is None:
		min_len = 2
	if max_len is None:
		max_len = timeseries.shape[1]

	if type(timeseries) == pd.DataFrame or type(timeseries) == pd.Series:
		timeseries = timeseries.values
	if type(labels) == pd.DataFrame or type(labels) == pd.Series:
		labels = labels.values

	max_gain, max_gap = 0, 0
	best_shapelet, best_dist, best_L = None, None, None
	for j in range(len(timeseries)):
		print('Extracting candidates from {}th timeseries'.format(j))
		# We will extract shapelet candidates from S
		S = timeseries[j, :]
		for l in range(min_len, max_len):  
			for i in range(len(S) - l + 1):
				candidate = S[i:i+l]
				# Compute distances to all other timeseries
				L = []  # The orderline, to calculate entropy
				for k in range(len(timeseries)):
					D = timeseries[k, :]
					dist = util.sdist(candidate, D)
					L.append((dist, labels[k]))
				L = sorted(L, key=lambda x: x[0])
				tau, updated, new_gain, new_gap = util.best_ig(L, max_gain, max_gap)
				if updated:
					best_shapelet = candidate
					print('Found new best shapelet of length {} with gain {} and gap {}'.format(len(best_shapelet), new_gain, new_gap))
					best_dist = tau
					best_L = L
					max_gain = new_gain
					max_gap = new_gap

	return best_shapelet, best_dist, best_L, max_gain, max_gap


