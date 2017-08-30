import pandas as pd
import numpy as np
from pyshapelets.shapelet_extraction.extract_shapelets import fit, extract_shapelet


TS_LENGTH = 50
prototype = [1] * TS_LENGTH

typical_characteristic = [2, 3, 4, 3, 2]
class_0 = [prototype] * 10
class_1 = []
time_series = []
labels = []

for i, timeseries in enumerate(class_0):
    # Put in the typical characteristic at random location
    k = np.random.randint(TS_LENGTH - len(typical_characteristic))
    ts = timeseries.copy()
    ts[k:len(typical_characteristic)+k] = typical_characteristic
    ts = np.array(ts) + (np.random.rand(TS_LENGTH) - 0.5)
    class_1.append(ts)

    class_0[i] = np.array(timeseries) + (np.random.rand(TS_LENGTH) - 0.5)

    time_series.append(ts)
    labels.append(1)
    time_series.append(class_0[i])
    labels.append(0)


labels = np.array(labels)
time_series = np.array(time_series)


import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(range(len(class_0[0])), class_0[0])
ax1.set_title('Class 0')
ax2.plot(range(len(class_1[0])), class_1[0])
ax2.set_title('Class 1')
plt.savefig('samples1.png')


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(range(len(class_0[2])), class_0[2])
ax1.set_title('Class 0')
ax2.plot(range(len(class_1[2])), class_1[2])
ax2.set_title('Class 1')
plt.savefig('samples2.png')

tree = extract_shapelet(time_series, labels)

for ts, label in zip(time_series, labels):
    print(label)
    print(tree.predict([ts]))
    print('-' * 100)