import time

from tqdm import tqdm

from pyshapelets.util import util
import numpy as np


def generate_binary_classification_data(typical_characteristic, TS_LENGTH=50, NR_TIMESERIES=100):
    prototype = [1] * TS_LENGTH
    class_0 = [prototype] * NR_TIMESERIES
    class_1 = []
    time_series = []
    labels = []

    for i, timeseries in enumerate(class_0):
        # Put in the typical characteristic at random location
        k = np.random.randint(TS_LENGTH - len(typical_characteristic))
        ts = timeseries.copy()
        ts[k:len(typical_characteristic) + k] = typical_characteristic
        ts = np.array(ts) + (np.random.rand(TS_LENGTH) - 0.5)
        class_1.append(ts)

        class_0[i] = np.array(timeseries) + (np.random.rand(TS_LENGTH) - 0.5)

        time_series.append(ts)
        labels.append(1)
        time_series.append(class_0[i])
        labels.append(0)

    return time_series, labels

typical_characteristic = [2, 3, 4, 3, 2]
ts_lengths = [10, 50]
nr_timeseries = [5, 10, 25, 50, 100]

for nr_timeserie in nr_timeseries:
    for ts_length in ts_lengths:
        print('Timing the distance calculation for', str(nr_timeserie), 'timeseries of length', str(ts_length))
        sdist_new_times = []
        sdist_new_overhead = []
        sdist_old_times = []
        timeseries, labels = generate_binary_classification_data(typical_characteristic, ts_length, nr_timeserie)
        for ts, label in tqdm(zip(timeseries, labels)):
            stats = {}
            start_time = time.time()
            for i, (ts2, label2) in enumerate(zip(timeseries, labels)):
                stats[tuple(ts2)] = util.calculate_stats(ts, ts2)
            sdist_new_overhead.append(time.time() - start_time)

            for l in range(1, ts_length + 1):
                for start in range(len(ts) - l):  # Possible start positions
                    new_dists = []
                    old_dists = []
                    for k, (ts2, label2) in enumerate(zip(timeseries, labels)):
                        start_time = time.time()
                        dist_new = util.sdist_new(ts[start:start + l], ts2, start, stats[tuple(ts2)])
                        sdist_new_times.append(time.time() - start_time)
                        new_dists.append((k, dist_new))

                        start_time = time.time()
                        dist_old, idx_old = util.subsequence_dist(ts2, ts[start:start + l])
                        sdist_old_times.append(time.time() - start_time)
                        old_dists.append((k, dist_old))

                    new_dists = sorted(new_dists, key=lambda x: x[1])
                    old_dists = sorted(old_dists, key=lambda x: x[1])
                    print(new_dists)
                    print(old_dists)
                    np.testing.assert_equal([x[0] for x in new_dists][0], [x[0] for x in old_dists][0])

        print('New distance calculation took:', np.sum(sdist_new_times) + np.sum(sdist_new_overhead))
        print('New distance (overhead):', np.sum(sdist_new_overhead))
        print('Old distance calculation took:', np.sum(sdist_old_times))


# New distance calculation took: 1.52813410759 for nr_timeserie = 5, length = 10
# New distance calculation took: 107.447799444 for nr_timeserie = 5, length = 50



