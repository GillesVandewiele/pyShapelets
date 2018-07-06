import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
from queue import Queue
from tqdm import trange

#TODO: Convert this to a class with fit- and predict- methods!

class LRUCache():
	def __init__(self, size=5):
		self.values = []
		self.size = size

	def put(self, value):
		while len(self.values) >= self.size:
			self.values.remove(self.values[0])

		self.values.append(value)


def extract_shapelet(timeseries, labels, min_len=None, max_len=None,
					 enable_pruning=False):
	# If no min_len and max_len are provided, we fill then in ourselves
	if min_len is None:
		min_len = 2
	if max_len is None:
		max_len = timeseries.shape[1]

	# Convert from pandas to numpy for HUGE speedups
	if type(timeseries) == pd.DataFrame or type(timeseries) == pd.Series:
		timeseries = timeseries.values
	if type(labels) == pd.DataFrame or type(labels) == pd.Series:
		labels = labels.values

	max_gain, max_gap = 0, 0
	best_shapelet, best_dist, best_L = None, None, None
	for j in trange(len(timeseries), desc='timeseries', position=0):
		print('Extracting candidates from {}th timeseries'.format(j))
		S = timeseries[j, :]
		stats = {}
		# Pre-compute all metric arrays, which will allow us to
		# calculate the distance between two timeseries in constant time
		for k in range(len(timeseries)):
			metrics = util.calculate_metric_arrays(S, timeseries[k, :])
			stats[(j, k)] = metrics

		for l in trange(min_len, max_len, desc='length', position=1):
			# Keep a history to calculate an upper bound, this could
			# result in pruning,LRUCache thus avoiding the construction of the
			# orderline L (which is an expensive operation)
			H = LRUCache(size=5)
			for i in range(len(S) - l + 1):
				if enable_pruning:
					# Check if we can prune
					prune = False
					for w in range(len(H.values)):
						L_prime, S_prime = H.values[w]
						R = util.sdist(S[i:i+l], S_prime)
						#print(R, util.upper_ig(L_prime.copy(), R), max_gain)
						if util.upper_ig(L_prime.copy(), R) < max_gain:
							#print('Prune at position {} with length {}'.format(i, l))
							prune = True
							break
					if prune: continue

				# Extract a shapelet from S, starting at index i with length l
				L = []  # An orderline with the distances to shapelet & labels
				for k in range(len(timeseries)):
					S_x, S_x2, S_y, S_y2, M = stats[(j, k)]
					L.append((
						util.sdist_metrics(i, l, S_x, S_x2, S_y, S_y2, M),
						labels[k]
					))
				L = sorted(L, key=lambda x: x[0])
				tau, updated, new_gain, new_gap = util.best_ig(L, max_gain, max_gap)
				if updated:
					best_shapelet = S[i:i+l]
					best_dist = tau
					best_L = L
					max_gain = new_gain
					max_gap = new_gap
				elif enable_pruning:
					H.put((L, S[i:i+l]))

	return best_shapelet, best_dist, best_L, max_gain, max_gap

