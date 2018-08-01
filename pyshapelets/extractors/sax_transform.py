import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import pickle
from scipy.stats import norm


def calculate_breakpoints(n):
	"""Calculate the n-1 different points in [-1, 1] which are the boundaries
	of different regions in N(0, 1) such that the probability of each
	region is equal."""
	split_points = []
	fraction = float(1)/float(n)
	for i in range(1, n):
		split_points.append(norm.ppf(i*fraction))
	return split_points


def calculate_distance_table(alphabet_size):
	"""Calculate a matrix of possible distances of two words in an alphabet
	resulting from a SAX-transformation"""
	break_points = calculate_breakpoints(alphabet_size)
	distance_table = np.zeros((alphabet_size, alphabet_size))
	for r in range(alphabet_size):
		for c in range(alphabet_size):
			if abs(r - c) <= 1:
				distance_table[r, c] = 0
			else:
				d1 = break_points[max(r, c) - 1]
				d2 = break_points[min(r, c)]
				distance_table[r, c] = d1 - d2
	return distance_table


def _paa(ts, nr_windows):
	window_means = []
	window_size = int(np.floor(float(len(ts)) / float(nr_windows)))
	for i in range(nr_windows - 1):
		window = ts[i*window_size:(i+1)*window_size]
		window_means.append(np.mean(window))
	window_means.append(np.mean(ts[i*window_size:]))
	return window_means


def sax_distance(x_sax, y_sax, ts_length, nr_windows, alphabet_size, 
				 distance_table=None):
	if distance_table is None:
		distance_table = calculate_distance_table(alphabet_size)
	
	total_distance = 0
	for x, y in zip(x_sax, y_sax):
		total_distance += distance_table[x, y]
	return np.sqrt(ts_length / nr_windows) * total_distance


def transform_ts(ts, nr_windows, alphabet_size, symbol_map):
	"""Transform a timeseries in their SAX representation"""
	#  if the standard deviation of the sequence before normalization is below 
	# an epsilon ε, we simply assign the entire word to 
	# the middle-ranged alphabet (e.g. 'cccccc' if a = 5)
	sequence = []
	window_means = _paa(ts, nr_windows)
	for mean in window_means:
		for interval in symbol_map:
			if interval[0] <= mean < interval[1]:
				sequence.append(symbol_map[interval])
	return np.array(sequence)


def get_symbol_map(alphabet_size):
	split_points = calculate_breakpoints(alphabet_size)
	symbol_map = {}
	symbol_map[(float('-inf'), split_points[0])] = 0
	for i, j in enumerate(range(len(split_points) - 1)):
		symbol_map[(split_points[j], split_points[j + 1])] = i + 1
	symbol_map[(split_points[-1], float('inf'))] = len(split_points)
	return symbol_map


def transform(timeseries, nr_windows, alphabet_size):
	"""Transform a collection of timeseries in their SAX representation"""
	symbol_map = get_symbol_map(alphabet_size)

	transformed_ts = [transform_ts(ts, nr_windows, alphabet_size, symbol_map) 
					  for ts in timeseries]
	return np.array(transformed_ts)
