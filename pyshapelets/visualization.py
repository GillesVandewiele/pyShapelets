import numpy as np
import matplotlib.pyplot as plt
import util
from collections import defaultdict


def visualize_shapelet(timeseries, labels, shapelet,
					   n_samples_per_class=3):
	"""Create a visualization by taking a few sample
	timeseries per class and then creating line graphs
	with both the timeseries and the shapelets (aligned
	at its best location)"""
	timeseries_per_class = defaultdict(list)
	unique_classes = set(labels)

	for ts, label in zip(timeseries, labels):
		timeseries_per_class[label].append(ts)

	random_samples = {}
	for c in unique_classes:
		random_idx = np.random.choice(
			range(len(timeseries_per_class[c])), 
			size=n_samples_per_class,
			replace=False
		)
		random_samples[c] = np.array(timeseries_per_class[c])[random_idx, :]

	f, ax = plt.subplots(n_samples_per_class, len(unique_classes))
	for i, c in enumerate(unique_classes):
		for j, ts in enumerate(random_samples[c]):
			dist, pos = util.sdist_with_pos(shapelet, ts)
			ax[j][i].plot(range(len(ts)), ts)
			ax[j][i].plot(range(pos, pos+len(shapelet)), shapelet)
			ax[j][i].set_title('Sample from class {} with distance {}'.format(c, dist))

	plt.show()
