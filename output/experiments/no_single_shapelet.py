import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
	+ os.sep 
	+ 'pyshapelets'
)
from extractors.brute_force import extract_shapelet
import util

# Define four time series
ts1 = [-1]*10 + [0]*5 + [1]*10
ts2 = [1]*10 + [0]*5 + [-1]*10

ts3 = [0]*15 + [1]*10
ts4 = [-1]*10 + [0]*15

X = np.array([ts1, ts2, ts3, ts4])
y = [0, 0, 1, 1]
# we cannot find a shapelet of max length 10 that gives
# perfect separation. Need for 
bad_shapelet, d, best_L, ig, g = extract_shapelet(X, y, max_len=11)
print(bad_shapelet)
print(best_L)

# This shapelet, which is not directly in the data
# can offer a perfect separation
good_shapelet = [-1]*2 + [1]*3 + [0]*5 + [-1]*3 + [1]*2
L = []
for k in range(len(X)):
	D = X[k, :]
	dist = util.sdist(good_shapelet, D)
	L.append((dist, y[k]))
print(L)

f, axarr = plt.subplots(5, sharex=True, sharey=True)
x = range(len(ts1))
axarr[0].plot(x, ts1, color='r')
axarr[1].plot(x, ts2, color='r')
axarr[2].plot(x, ts3)
axarr[3].plot(x, ts4)
axarr[4].plot(range(len(good_shapelet)), good_shapelet)
plt.show()