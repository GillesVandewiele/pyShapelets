import pandas as pd
import numpy as np
from extract_shapelets import extract_shapelet


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
data = pd.read_csv('data/Coffee.csv', header=None)
# data = data.sample(20)

# Set the column names
feature_cols = []
for i in range(286): feature_cols.append('x_'+str(i))
data.columns = feature_cols + ['label']
print(len(data))

# pre-pruning:

# no pre-pruning:
# Brute force algorithm took 114.40455603599548
# 0.12408303257700161 per iteration

# data = pd.read_csv('data/Beef.csv', header=None)
# # data = data.sample(20)
#
# # Set the column names
# feature_cols = []
# for i in range(470): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']

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

extract_shapelet(timeseries, labels, 50, 50)
