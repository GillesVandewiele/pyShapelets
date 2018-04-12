import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util

#TODO: Convert this to a class with fit- and predict- methods!


def extract_shapelet(timeseries, labels):
	m = timeseries.shape[1]
	max_gain, max_gap = 0, 0
	for j in range(len(timeseries)):
		S = timeseries.iloc[j, :]
		stats = {}
		for k in range(len(timeseries)):
			metrics = util.calculate_metric_arrays(S, timeseries.iloc[k, :])
			stats[(i, k)] = metrics

		for l in range(1, m):
			pass

