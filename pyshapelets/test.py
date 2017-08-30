import pandas as pd
import numpy as np
from pyshapelets.shapelet_extraction.extract_shapelets import fit, extract_shapelet

# def adjustProbabilities(probs):
#     for i in range(len(probs)):
#         j = np.random.choice(len(probs), size=1)[0]
#         p, p2 = probs[i], probs[j]
#         mut = np.random.uniform(-min(p, p2), min(p, p2), size=1)[0]
#         probs[i] += mut
#         probs[j] -= mut
#     return probs
#
# print(adjustProbabilities([0.1,0.1,0.1,0.5,0.1,0.1]))

# # Read in the coffee dataset
from pyshapelets.util.util import calculate_stats, subsequence_dist, subsequence_dist_z_space, sdist_new
from pyshapelets.util.util import calculate_stats_old
from pyshapelets.visualization.visualization import run

# data = pd.read_csv('data/Coffee.csv', header=None)
# # data = data.sample(20)
#
# # Set the column names
# feature_cols = []
# for i in range(286): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']
# print(len(data))

# pre-pruning:

# no pre-pruning:
# Brute force algorithm took 114.40455603599548
# 0.12408303257700161 per iteration

data = pd.read_csv('data/Beef.csv', header=None)
data = data.sample(25 , random_state=1337)

# Set the column names
feature_cols = []
for i in range(470): feature_cols.append('x_'+str(i))
data.columns = feature_cols + ['label']

# data = data[data['label'] > 0]

# data = pd.read_csv('data/Wine.csv', header=None)
# # data = data.sample(20)
#
# # Set the column names
# feature_cols = []
# for i in range(234): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']
# from collections import Counter
# print(Counter(data['label']))

# data = pd.read_csv('data/ArrowHead.csv', header=None)
# feature_cols = []
# for i in range(251): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']
# print(len(data))

# Split in timeseries and labels
labels = data['label'].values
timeseries = data.drop('label', axis=1).values

# m_uv = np.around(calculate_stats(timeseries[0], timeseries[1])[4], 7)
# m_uv_old = np.around(calculate_stats_old(timeseries[0], timeseries[1])[4], 7)
#
# print(m_uv.shape)
# print(m_uv_old.shape)
#
# for i in range(m_uv.shape[0]):
#     for j in range(m_uv.shape[1]):
#         if float(m_uv[i, j]) != float(m_uv_old[i, j]):
#             print(i, j, m_uv[i,j], m_uv_old[i,j])
#
# assert np.array_equal(m_uv, m_uv_old)

print(labels)
print(timeseries)

if __name__ == "__main__":
    print('Fitting tree')
    tree = extract_shapelet(timeseries, labels)

    print(tree.shapelet)
    print(tree.distance)

    distances = []

    for ts, label in zip(timeseries, labels):
        d, idx = subsequence_dist(ts, tree.shapelet)
        distances.append((d, label))

    print([x for x in sorted(distances, key=lambda x: x[0])])

    distances = []

    for ts, label in zip(timeseries, labels):
        stats = calculate_stats(tree.shapelet, ts)
        d = sdist_new(tree.shapelet, ts, 0, stats)
        distances.append((d, label))

    print([x for x in sorted(distances, key=lambda x: x[0])])

    distances = []

    for ts, label in zip(timeseries, labels):
        stats = calculate_stats(tree.right.shapelet, ts)
        d = sdist_new(tree.right.shapelet, ts, 0, stats)
        distances.append((d, label))

    print([x for x in sorted(distances, key=lambda x: x[0])])


    # tree.populate_class_probs(timeseries[:-1], labels[:-1])
    # tree.recalculate_distances(timeseries[:-1], labels[:-1])

    for ts, label in zip(timeseries, labels):
        print(label)
        print(tree.predict([ts], z_norm=True))
        print('-'*100)

    # tree = fit(timeseries, labels, max_len=50, min_len=50)
    # print('Creating visualization')
    # run(tree)
