import pandas as pd

from extract_shapelets import extract_shapelet


# # Read in the coffee dataset
# data = pd.read_csv('data/Coffee.csv', header=None)
# # data = data.sample(20)
#
# # Set the column names
# feature_cols = []
# for i in range(286): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']
# print(len(data))

# data = pd.read_csv('data/Beef.csv', header=None)
# # data = data.sample(20)
#
# # Set the column names
# feature_cols = []
# for i in range(470): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']

data = pd.read_csv('data/Wine.csv', header=None)
# data = data.sample(20)

# Set the column names
feature_cols = []
for i in range(234): feature_cols.append('x_'+str(i))
data.columns = feature_cols + ['label']
from collections import Counter
print(Counter(data['label']))

# data = pd.read_csv('data/ArrowHead.csv', header=None)
# feature_cols = []
# for i in range(251): feature_cols.append('x_'+str(i))
# data.columns = feature_cols + ['label']
# print(len(data))

# Split in timeseries and labels
labels = data['label'].values
timeseries = data.drop('label', axis=1).values

extract_shapelet(timeseries, labels, 50, 50)
