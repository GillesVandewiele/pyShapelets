import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from algorithms import ShapeletTransformer
from extractors.extractor import MultiGeneticExtractor
from data.load_all_datasets import load_data_train_test

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from tslearn.shapelets import ShapeletModel


np.random.seed(1337)
random.seed(1337)

TRAIN_PATH = 'data/partitioned/MoteStrain/MoteStrain_train.csv'
TEST_PATH = 'data/partitioned/MoteStrain/MoteStrain_test.csv'

# Load the training and testing dataset (features + label vector)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target']
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target']

map_dict = {}
for j, c in enumerate(np.unique(y_train)):
    map_dict[c] = j
y_train = y_train.map(map_dict) 
y_test = y_test.map(map_dict)

print(set(y_train), set(y_test))

y_train = y_train.values
y_test = y_test.values

measurements = []

for pop_size in [5, 25, 50, 100, 250, 500, 1000]:
    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=pop_size, iterations=50, wait=10, plot=False)
    shapelets = genetic_extractor.extract(X_train, y_train)
    shap_transformer = ShapeletTransformer()
    shap_transformer.shapelets = shapelets

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[POP SIZE]', pop_size, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([pop_size, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of Genetic Algorithm on hyper-parameter pop. size')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['popsize', 'accuracy'])
measurements_df.to_csv('gen_param_pop_size.csv')


for iterations in [5, 25, 50, 100, 250]:
    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=50, iterations=iterations, wait=10, plot=False)
    shapelets = genetic_extractor.extract(X_train, y_train)
    shap_transformer = ShapeletTransformer()
    shap_transformer.shapelets = shapelets

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[ITERATIONS]', iterations, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([iterations, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of Genetic Algorithm on hyper-parameter iterations')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['iterations', 'accuracy'])
measurements_df.to_csv('gen_param_iterations.csv')

for mutation_prob in [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]:
    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=50, iterations=50, wait=10, plot=False, add_noise_prob=mutation_prob, add_shapelet_prob=mutation_prob, remove_shapelet_prob=mutation_prob)
    shapelets = genetic_extractor.extract(X_train, y_train)
    shap_transformer = ShapeletTransformer()
    shap_transformer.shapelets = shapelets

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[MUT PROB]', mutation_prob, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([mutation_prob, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of Genetic Algorithm on hyper-parameter mut. prob.')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['mut. prob.', 'accuracy'])
measurements_df.to_csv('gen_param_mut_prob.csv')


for crossover_prob in [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]:
    genetic_extractor = MultiGeneticExtractor(verbose=True, population_size=50, iterations=50, wait=10, plot=False, crossover_prob=crossover_prob)
    shapelets = genetic_extractor.extract(X_train, y_train)
    shap_transformer = ShapeletTransformer()
    shap_transformer.shapelets = shapelets

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)

    print('[CX PROB]', crossover_prob, accuracy_score(y_test, lr.predict(X_distances_test)))

    measurements.append([crossover_prob, accuracy_score(y_test, lr.predict(X_distances_test))])

plt.figure()
plt.plot([x[0] for x in measurements], [x[1] for x in measurements])
plt.title('Sensitive of Genetic Algorithm on hyper-parameter cx. prob.')
plt.show()

measurements_df = pd.DataFrame(measurements, columns=['cx. prob.', 'accuracy'])
measurements_df.to_csv('gen_param_cx_prob.csv')