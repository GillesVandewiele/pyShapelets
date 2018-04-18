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

# TODO: Write some

def test_sax_distance():
	np.random.seed(1337)
	TS_LENGTH = 100
	WINDOWS = 10
	ALPHABET_SIZE = 8
	ts1 = np.random.normal(0, 1, size=100)
	ts2 = ts1 + np.random.rand(100)/10
	ts3 = np.random.normal(0, 1, size=100)
	transformed_ts = transform([ts1, ts2, ts3], WINDOWS, ALPHABET_SIZE)
	dist1 = sax_distance(transformed_ts[0], transformed_ts[1], TS_LENGTH,
						 WINDOWS, ALPHABET_SIZE) 
	dist2 = sax_distance(transformed_ts[0], transformed_ts[2], TS_LENGTH,
						 WINDOWS, ALPHABET_SIZE)
	dist3 = sax_distance(transformed_ts[1], transformed_ts[2], TS_LENGTH,
						 WINDOWS, ALPHABET_SIZE)
	print(transformed_ts[0], transformed_ts[1], transformed_ts[2])
	print(dist1, dist2, dist3)
	assert dist1 < dist2
	assert dist1 < dist3

#test_sax_distance()

"""
print(calculate_breakpoints(10))
print(calculate_distance_table(4))
input()

import matplotlib.pyplot as plt

TS_LENGTH = 100
timeseries = np.zeros((2, 100))
for i in range(timeseries.shape[0]):
	timeseries[i, :] = util.z_norm(np.random.rand(TS_LENGTH))


WINDOWS = 10
WINDOW_SIZE = int(np.ceil(TS_LENGTH / WINDOWS))
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(range(100), timeseries[0, :], color='b')
paa_transform = _paa(timeseries[0, :], WINDOWS)
transformed_ts = []
for mean in paa_transform:
	transformed_ts += [mean] * WINDOW_SIZE
ax1.plot(range(100), transformed_ts, color='r')
ax2.plot(range(100), timeseries[1, :], color='b')
paa_transform = _paa(timeseries[1, :], WINDOWS)
transformed_ts = []
for mean in paa_transform:
	transformed_ts += [mean] * WINDOW_SIZE
ax2.plot(range(100), transformed_ts, color='r')
plt.show()

# TODO: check what to do if length of time series is not divisble
# by number of windows. In the paper, they propose to account some
# points for only a fraction of their weight. 
# e.g.: t0 t1 t2 t3 t4 t5 t6 (3 windows)
#       ------ ------ ------
# --> t2 is divided (1/3th in first window, 2/3th in second)
# --> t4 is divided (2/3th in second window, 1/3th in first)
# (But seems rather strange )
transformed_timeseries = transform(timeseries, WINDOWS, 8)
print(sax_distance(transformed_timeseries[0], transformed_timeseries[1],
				   TS_LENGTH, WINDOWS, 8))
"""