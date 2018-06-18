# Standard library
from collections import defaultdict, Counter
import array
import time

# pip-installable libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from deap import base, creator, algorithms, tools
from deap.benchmarks.tools import diversity
#from tslearn.shapelets import ShapeletModel

# Some beautiful Python imports
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sax_transform import transform
import util

#from sklearn.metrics.cluster import silhouette_score as sklearn_silhouette_score
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.clustering import GlobalAlignmentKernelKMeans, silhouette_score
from tslearn.barycenters import euclidean_barycenter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import set_config

from pathos.multiprocessing import ProcessingPool as Pool

import array

from mstamp.mstamp_stomp import mstamp as mstamp_stomp


def extract_top_k_shapelets(shapelets, k, sort_key):
    """Extract the top-k shapelets. First sort them, according to their
    sort_key (the quality), then remove 'similar' shapelets which are
    extracted from the same timeseries and have overlapping indices

    Shapelets should be a list:
        [
            [shapelet (list), ts_idx (int), start_idx (int), length (int)], 
            [score (list)]
        ]
    """ 
    shapelets = [[x[0]] + x[-3:] for x in sorted(shapelets, key=sort_key)]
    best_shapelets = defaultdict(list)
    for shapelet, ts_idx, start_idx, length in shapelets:
        if sum([len(x) for x in best_shapelets.values()]) == k:
            break

        if len(best_shapelets[ts_idx]) == 0:
            best_shapelets[ts_idx].append([shapelet, ts_idx, start_idx, length])
        else:
            add = True
            for shapelet2, _, start_idx2, length2 in best_shapelets[ts_idx]:
                if start_idx in range(start_idx2, start_idx2+length2):
                    add = False
                    break
            if add:
                best_shapelets[ts_idx].append([shapelet, ts_idx, start_idx, length])

    top_k_shapelets = []
    for key in best_shapelets:
        for shapelet, _, _, _ in best_shapelets[key]:
            top_k_shapelets.append(shapelet)
    return top_k_shapelets


class Extractor(object):
    def __init__(self):
        pass

    def _convert_to_numpy(self, timeseries, labels):
        try:
            if (type(timeseries) == pd.DataFrame 
                or type(timeseries) == pd.Series):
                timeseries = timeseries.values
            for ts_idx, ts in enumerate(timeseries):
                if type(ts) == list:
                    timeseries[ts_idx] = np.array(ts)
            if type(timeseries) == list:
                timeseries = np.array(timeseries)

            if type(labels) == pd.DataFrame or type(labels) == pd.Series:
                labels = labels.values
            if type(labels) == list:
                labels = np.array(labels)

            return timeseries, labels
        except:
            print('exception occurred')
            pass

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        self.timeseries, self.labels = self._convert_to_numpy(timeseries, 
                                                              labels)

        # If no min_len and max_len are provided, we initialise them
        if min_len is None:
            min_len = 1
        if max_len is None:
            max_len = self.timeseries.shape[1]

        self.nr_shapelets = nr_shapelets
        self.min_len = min_len
        self.max_len = max_len

        self.metric = {
            'ig': util.calculate_ig,
            'kw': util.kruskal_score,
            'f': util.f_score,
            'mm': util.mood_median,
        }[metric]
        
        self.key = {
            'ig': lambda x: (-x[1], -x[2]),
            'kw': lambda x: -x[1],
            'f': lambda x: -x[1],
            'mm': lambda x: -x[1]
        }[metric]


class BruteForceExtractor(Extractor):
    def __init__(self):
        pass

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(BruteForceExtractor, self).extract(timeseries, labels, min_len,
                                                 max_len, nr_shapelets, metric)
        shapelets = []
        for j in trange(len(self.timeseries), desc='timeseries', position=0):
            # We will extract shapelet candidates from S
            S = self.timeseries[j, :]
            for l in range(self.min_len, self.max_len):  
                for i in range(len(S) - l + 1):
                    candidate = S[i:i+l]
                    # Compute distances to all other timeseries
                    L = []  # The orderline, to calculate entropy
                    for k in range(len(self.timeseries)):
                        D = self.timeseries[k, :]
                        dist = util.sdist(candidate, D)
                        L.append((dist, self.labels[k]))
                    score = self.metric(L)
                    shapelets.append(([list(candidate)] + list(score) + [j, i, l]))

        shapelets = sorted(shapelets, key=self.key)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets


class LRUCache():
    def __init__(self, size=5):
        self.values = []
        self.size = size

    def put(self, value):
        while len(self.values) >= self.size:
            self.values.remove(self.values[0])

        self.values.append(value)


class FastExtractor(Extractor):
    def __init__(self, pruning=False, cache_size=10):
        self.pruning = pruning
        self.cache_size = cache_size

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(FastExtractor, self).extract(timeseries, labels, min_len,
                                           max_len, nr_shapelets, metric)
        shapelets = []
        for j in trange(len(self.timeseries), desc='timeseries', position=0):
            S = self.timeseries[j, :]
            stats = {}
            # Pre-compute all metric arrays, which will allow us to
            # calculate the distance between two timeseries in constant time
            for k in range(len(self.timeseries)):
                metrics = util.calculate_metric_arrays(S, self.timeseries[k, :])
                stats[(j, k)] = metrics

            for l in range(self.min_len, self.max_len):  
                # Keep a history to calculate an upper bound, this could
                # result in pruning,LRUCache thus avoiding the construction of the
                # orderline L (which is an expensive operation)
                H = LRUCache(size=self.cache_size)
                for i in range(len(S) - l + 1):
                    if self.pruning:
                        # Check if we can prune
                        prune = False
                        for w in range(len(H.values)):
                            L_prime, S_prime = H.values[w]
                            R = util.sdist(S[i:i+l], S_prime)
                            if util.upper_ig(L_prime.copy(), R) < max_gain:
                                prune = True
                                break
                        if prune: continue

                    # Extract a shapelet from S, starting at index i with length l
                    L = []  # An orderline with the distances to shapelet & labels
                    for k in range(len(self.timeseries)):
                        S_x, S_x2, S_y, S_y2, M = stats[(j, k)]
                        L.append((
                            util.sdist_metrics(i, l, S_x, S_x2, S_y, S_y2, M),
                            self.labels[k]
                        ))
                    score = self.metric(L)
                    shapelets.append(([list(S[i:i+l])] + list(score) + [j, i, l]))

                    if self.pruning:
                        H.put((L, S[i:i+l]))

        shapelets = sorted(shapelets, key=self.key)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets
        

class LearningExtractor(Extractor):
    # TODO: create a dictionary, with keys=[min_len, max_len] and
    # the values are equal to nr_shapelets

    # After extracting them with tslearn, iterate over them and create a 
    # top K
    def __init__(self, batch_size=4, max_iter=50, weight_regularizer=0.0,
                 optimizer='sgd'):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.weight_regularizer = weight_regularizer
        self.optimizer = optimizer
        #from numpy.random import seed
        #seed(1)
        #from tensorflow import set_random_seed
        #set_random_seed(2)

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(LearningExtractor, self).extract(timeseries, labels, min_len,
                                               max_len, nr_shapelets, metric)

        all_extracted_shapelets = []
        for i in trange(self.min_len, self.max_len, desc='length', position=0):
            # IMPORTANT! This parameter has an impact on the output
            shapelet_sizes= {i: self.nr_shapelets}

            clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes, 
                                max_iter=self.max_iter, verbose_level=0,
                                optimizer=self.optimizer,
                                weight_regularizer=self.weight_regularizer)
            extracted_shapelets = clf.fit(
                np.reshape(
                    self.timeseries, 
                    (self.timeseries.shape[0], self.timeseries.shape[1], 1)
                ), 
                self.labels
            ).shapelets_
            for shapelet in extracted_shapelets:
                all_extracted_shapelets.append(shapelet)


        shapelets = []
        for candidate in all_extracted_shapelets:
            L = []  # The orderline, to calculate entropy
            for k in range(len(self.timeseries)):
                D = self.timeseries[k, :]
                dist = util.sdist(candidate, D)
                L.append((dist, self.labels[k]))
            score = self.metric(L)
            shapelets.append(([list(candidate)]+list(score)))

        shapelets = sorted(shapelets, key=self.key)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets


class MultiGeneticExtractor(Extractor):
    def __init__(self, population_size=25, iterations=50, verbose=True,
                 add_noise_prob=0.2, add_shapelet_prob=0.2, 
                 remove_shapelet_prob=0.2, crossover_prob=0.5, wait=10,
                 plot=True):
        """

        """
        np.random.seed(1337)
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.add_noise_prob = add_noise_prob
        self.add_shapelet_prob = add_shapelet_prob
        self.remove_shapelet_prob = remove_shapelet_prob
        self.crossover_prob = crossover_prob
        self.plot = plot
        self.wait = wait


    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        """

        """
        # Call the super-class in order to initialize all required variables
        super(MultiGeneticExtractor, self).extract(timeseries, labels, min_len,
                                                   max_len, nr_shapelets, 
                                                   metric)

        # We will try to maximize the class-scatter matrix score.
        # In the case of ties, we pick the one with least number of shapelets
        weights = (1.0, -1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)

        # Individual are lists (of shapelets (list))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def random_shapelet():
            """Extract a random subseries from the training set"""
            rand_row = np.random.randint(self.timeseries.shape[0])
            rand_length = np.random.randint(self.min_len, self.max_len)
            rand_col = np.random.randint(self.timeseries.shape[1] - rand_length)
            return list(self.timeseries[rand_row, rand_col:rand_col+rand_length])

        def create_individual():
            if np.random.random() < 0.25:
                return [random_shapelet()]
            else:
                # Seed the population with some motifs
                rand_length = np.random.randint(self.min_len, self.max_len)
                subset_idx = np.random.choice(range(len(self.timeseries)), 
                                              size=int(0.5*len(self.timeseries)), 
                                              replace=False)
                ts = self.timeseries[subset_idx, :].flatten()
                matrix_profile, _ = mstamp_stomp(ts, rand_length)
                motif_idx = matrix_profile[0, :].argsort()[-1]
                return [ts[motif_idx:motif_idx + rand_length]]

        def cost(shapelets):
            """Evaluate an individual, composed of multiple shapelets. First,
            construct a (|samples| x |shapelets|) matrix with distances 
            from each sample to each shapelet. Then, construct the class
            scatter matrix (used in LDA) based on the labels.
            """
            start = time.time()
            X = np.zeros((len(self.timeseries), len(shapelets)))
            for k in range(len(self.timeseries)):
                D = self.timeseries[k, :]
                for j in range(len(shapelets)):
                    dist = util.sdist(shapelets[j], D)
                    X[k, j] = dist

            lr = LogisticRegression()
            lr.fit(X, self.labels)
            return (np.mean(cross_val_score(lr, X, self.labels, cv=3, scoring='f1_micro')), sum([len(x) for x in shapelets]))#neg_log_loss, f1_micro
            #return (lr.score(X, self.labels), len(shapelets))

            #return (util.class_scatter_matrix(X, self.labels), sum([len(x) for x in shapelets]))

        def add_noise(shapelets):
            """Add random noise to a random shapelet"""
            rand_shapelet = np.random.randint(len(shapelets))
            tools.mutGaussian(shapelets[rand_shapelet], 
                              mu=0, sigma=0.25, indpb=0.25)

            return shapelets,

        def add_shapelet(shapelets):
            """Add a random shapelet to the individual"""
            shapelets.append(random_shapelet())

            return shapelets,

        def remove_shapelet(shapelets):
            """Remove a random shapelet from the individual"""
            if len(shapelets) > 1:
                rand_shapelet = np.random.randint(len(shapelets))
                shapelets.pop(rand_shapelet)

            return shapelets,

        def crossover(ind, shapelet_to_cluster, shapelets_per_cluster):
            # First, take a random shapelet from the individual,
            # then select a shapelet from the same cluster,
            # using roulette selection, and then apply randomly
            # either one-point (cxOnePoint) or two-point (cxTwoPoint) crossover.
            rand_idx = np.random.randint(len(ind))
            rand_shapelet = ind[rand_idx]
            cluster = shapelet_to_cluster[tuple(rand_shapelet)]
            similar_shapelets = shapelets_per_cluster[cluster]

            if len(similar_shapelets) <= 1:
                return ind

            #TODO: Can we replace this by picking randomly according to shapelet quality?
            other_shapelet = similar_shapelets[np.random.choice(range(len(similar_shapelets)))]  

            if np.random.random() < 0.5:
                offspring = tools.cxOnePoint(list(rand_shapelet).copy(), list(other_shapelet).copy())[0]
            else:
                offspring = tools.cxTwoPoint(list(rand_shapelet).copy(), list(other_shapelet).copy())[0]

            ind[rand_idx] = offspring

            return ind,

        def crossover2(ind1, ind2):
            similarity_matrix = cdist_gak(ind1, ind2)#, sigma=sigma_gak(list(ind1)+list(ind2)))
            for row_idx in range(similarity_matrix.shape[0]):
                non_equals = similarity_matrix[row_idx, :][similarity_matrix[row_idx, :] != 1.0]
                if len(non_equals):
                    max_col_idx = np.argmax(similarity_matrix[row_idx, :][similarity_matrix[row_idx, :] != 1.0])
                    ind1[row_idx] = euclidean_barycenter([list(ind1[row_idx]).copy(), list(ind2[max_col_idx]).copy()])
                    ind1[row_idx] = ind1[row_idx][~np.isnan(ind1[row_idx])]
                    #if np.random.random() < 0.5:
                    #    ind1[row_idx] = tools.cxOnePoint(list(ind1[row_idx]).copy(), list(ind2[max_col_idx]).copy())[0]
                    #else:
                    #    ind1[row_idx] = tools.cxTwoPoint(list(ind1[row_idx]).copy(), list(ind2[max_col_idx]).copy())[0]

            for col_idx in range(similarity_matrix.shape[1]):
                non_equals = similarity_matrix[:, col_idx][similarity_matrix[:, col_idx] != 1.0]
                if len(non_equals):
                    max_row_idx = np.argmax(similarity_matrix[:, col_idx])
                    ind2[col_idx] = euclidean_barycenter([list(ind1[max_row_idx]).copy(), list(ind2[col_idx]).copy()])
                    ind2[col_idx] = ind2[col_idx][~np.isnan(ind2[col_idx])]
                    #if np.random.random() < 0.5:
                    #    ind2[col_idx] = tools.cxOnePoint(list(ind1[max_row_idx]).copy(), list(ind2[col_idx]).copy())[0]
                    #else:
                    #    ind2[col_idx] = tools.cxTwoPoint(list(ind1[max_row_idx]).copy(), list(ind2[col_idx]).copy())[0]

            return ind1, ind2

        def crossover3(ind1, ind2):
            if len(ind1) > 1 and len(ind2) > 1:
                if np.random.random() < 0.5:
                    ind1, ind2 = tools.cxOnePoint(list(ind1), list(ind2))
                else:
                    ind1, ind2 = tools.cxTwoPoint(list(ind1), list(ind2))
            
            return ind1, ind2

        set_config(assume_finite=True)

        # Register all operations in the toolbox
        toolbox = base.Toolbox()

        # TODO: comment this out and check if it doesnt run faster bcs of MP overhead
        pool = Pool(4)
        toolbox.register("map", pool.map)

        toolbox.register("mate", crossover2)
        toolbox.register("mate2", crossover3)
        toolbox.register("mutate", add_noise)
        toolbox.register("add", add_shapelet)
        toolbox.register("remove", remove_shapelet)

        toolbox.register("individual",  tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", cost)
        # Small tournaments to ensure diversity
        toolbox.register("select", tools.selTournament, tournsize=3)#tools.selSPEA2)#tools.selDoubleTournament, fitness_size=3,
                         #parsimony_size=1.5, fitness_first=True) #selNSGA2

        # Set up the statistics. We will measure the mean, std dev and 
        # maximum value of the class-scatter-matrix score.
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)
        if self.verbose:
            print('it\t\tavg\t\tstd\t\tmax\t\ttime')

        # Initialize the population and calculate their initial fitness values
        pop = toolbox.population(n=self.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Keep track of the best iteration, in order to do stop after `wait`
        # generations without improvement
        it, best_it = 1, 1
        best_ind = None
        best_score = float('-inf')

        # Set up a matplotlib figure and set the axes
        height = int(np.ceil(self.population_size/4))
        if self.plot:
            if self.population_size <= 20:
                f, ax = plt.subplots(4, height, sharex=True)
            else:
                plt.figure(figsize=(15, 5))
                plt.xlim([0, len(timeseries[0])])

        # The genetic algorithm starts here
        while it <= self.iterations and it - best_it < self.wait:
            gen_start = time.time()

            """
            # Apply clustering on all shapelets of each individual in population
            shapelet_to_cluster = {}
            shapelets_per_cluster = defaultdict(list)
            all_shapelets = []
            for ind in pop:
                for shapelet in ind:
                    all_shapelets.append(np.array(shapelet, dtype=np.float))

            all_shapelets = np.array(all_shapelets, dtype=np.ndarray)

            best_c_score = float('-inf')
            best_nr_clusters = 0
            
            start = time.time()
            for i in range(2, int(np.ceil(np.sqrt(len(all_shapelets)))) + 1):
                tsk = GlobalAlignmentKernelKMeans(n_clusters=i, verbose=False, sigma=sigma_gak(all_shapelets))
                labels = tsk.fit_predict(all_shapelets)
                set_config(assume_finite=True)
                score = silhouette_score(all_shapelets, labels)
                if score > best_c_score:
                    best_nr_clusters = i
                    best_c_score = score
            end = time.time()

            print('Population of {} individuals and {} shapelets divided into {} clusters (Took {}s)'.format(len(pop), len(all_shapelets), best_nr_clusters, end-start))

            tsk = GlobalAlignmentKernelKMeans(n_clusters=best_nr_clusters, verbose=False, sigma=sigma_gak(all_shapelets))
            labels = tsk.fit_predict(all_shapelets)
            for shapelet, label in zip(all_shapelets, labels):
                shapelet_to_cluster[tuple(shapelet)] = label
                shapelets_per_cluster[label].append(shapelet)
            """

            # Apply selection
            start = time.time()
            #offspring = toolbox.select(pop, self.population_size)
            offspring = list(map(toolbox.clone, pop))
            #print('Selection took {}s'.format(time.time() - start))

            # Plot the fittest individual of our population
            if self.plot:
                if self.population_size <= 20:
                    #f, ax = plt.subplots(4, height, sharex=True)
                    #plt.clf()
                    for ix, ind in enumerate(offspring):
                        ax[ix//height][ix%height].clear()
                        for shap in ind:
                            ax[ix//height][ix%height].plot(range(len(shap)), shap)
                    plt.pause(0.001)

                else:
                    best_ind = pop[np.argmax([x.fitness.values[0] for x in pop])]
                    plt.clf()
                    for shap in best_ind:
                        plt.plot(range(len(shap)), shap)
                    plt.pause(0.001)

            # Iterate over all individuals and apply cross-over with certain prob
            start = time.time()
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                try:
                    if np.random.random() < self.crossover_prob:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                    if np.random.random() < self.crossover_prob:
                        toolbox.mate2(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                except:
                    pass
            #print('Crossovers took {}s'.format(time.time() - start))

            # Apply mutation to each individualnoise_prob:
            start = time.time()
            for idx, indiv in enumerate(offspring):
                if np.random.random() < self.add_noise_prob:
                    toolbox.mutate(indiv)
                    del indiv.fitness.values
                if np.random.random() < self.add_shapelet_prob:
                    toolbox.add(indiv)
                    del indiv.fitness.values
                if np.random.random() < self.remove_shapelet_prob:
                    toolbox.remove(indiv)
                    del indiv.fitness.values
            #print('Mutations took {}s'.format(time.time() - start))

            # Update the fitness values            
            start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            #print('Calculating fitnesses took {}s'.format(time.time() - start))

            # Replace population and update hall of fame & statistics
            start = time.time()
            pop[:] = toolbox.select(offspring, self.population_size - 1) + tools.selBest(pop + offspring, 1)
            it_stats = stats.compile(pop)
            #print('Stats took {}s'.format(time.time() - start))

            # Print our statistics
            if self.verbose:
                print('{}\t\t{}\t\t{}\t\t{}\t{}'.format(
                    it, 
                    np.around(it_stats['avg'], 4), 
                    np.around(it_stats['std'], 3), 
                    np.around(it_stats['max'], 6),
                    np.around(time.time() - gen_start, 4), 
                ))

            # The smaller the value is, the better the front is.
            #try:
            #    print('Pop diversity:', diversity(pop, sorted(pop, key=lambda x: (x.fitness.values[0], -x.fitness.values[1]))[0].fitness.values, 
            #          sorted(pop, key=lambda x: (x.fitness.values[0], -x.fitness.values[1]))[-1].fitness.values))
            #except:
            #    print('Calc diversity did not work')

            # Have we found a new best score?
            if it_stats['max'] > best_score:
                best_it = it
                best_score = it_stats['max']
                best_ind = pop[np.argmax([x.fitness.values[0] for x in pop])].copy()

            it += 1

        return best_ind


class GeneticExtractor(Extractor):
    # TODO: Implement co-evolution, where we evolve multiple populations.
    # TODO: One population per specified length. Else, the population converges
    # TODO: to similar shapelets of same length. Or alternatively: speciation!
    def __init__(self, population_size=25, iterations=50, verbose=True,
                 mutation_prob=0.25, crossover_prob=0.25, wait=5):
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.wait = wait
        np.random.seed(1337)


    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        # TODO: If nr_shapelets > 1, then represent individuals by
        # TODO: `nr_shapelets` shapelets (instead of taking top-k from hof)
        super(GeneticExtractor, self).extract(timeseries, labels, min_len,
                                              max_len, nr_shapelets, metric)

        weights = (1.0,)
        if metric == 'ig':
            weights = (1.0, 1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax, score=None)

        def random_shapelet():
            rand_row_idx = np.random.randint(self.timeseries.shape[0])
            rand_length = np.random.choice(range(self.min_len, self.max_len), size=1)[0]
            rand_col_start_idx = np.random.randint(self.timeseries.shape[1] - rand_length)
            return self.timeseries[
                rand_row_idx, 
                rand_col_start_idx:rand_col_start_idx+rand_length
            ]

        def cost(shapelet):
            L = []
            for k in range(len(self.timeseries)):
                D = self.timeseries[k, :]
                dist = util.sdist(shapelet, D)
                L.append((dist, self.labels[k]))
            return self.metric(L)

        def mutation(pcls, shapelet):
            if np.random.random() < self.mutation_prob:
                tools.mutGaussian(shapelet, mu=0, sigma=0.1, indpb=0.1)[0]

            if np.random.random() < self.mutation_prob:
                rand_idx = np.random.randint(len(shapelet))
                del shapelet[rand_idx]

            if np.random.random() < self.mutation_prob:
                rand_idx = np.random.randint(len(shapelet))
                rand_elt = np.random.rand() * 2 - 1
                shapelet.insert(rand_idx, rand_elt)


        toolbox = base.Toolbox()
        toolbox.register("mate_one", tools.cxOnePoint)
        toolbox.register("mate_two", tools.cxTwoPoint)
        toolbox.register("mutate", mutation, creator.Individual)

        toolbox.register("individual",  tools.initIterate, creator.Individual, random_shapelet)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", cost)
        toolbox.register("select", tools.selRoulette)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        hof = tools.HallOfFame(nr_shapelets)

        pop = toolbox.population(n=self.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        it, best_it = 1, 1
        best_score = float('-inf')
        print('it\t\tavg\t\tstd\t\tmax\t\ttime')

        while it <= self.iterations and it - best_it < self.wait:
            #print(Counter([len(x) for x in pop]))
            start = time.time()

            # Apply selection and cross-over the selected individuals
            # TODO: Move this to a cross-over for-loop and just select 2 individuals in each it
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    if np.random.random() < 0.5:
                        toolbox.mate_one(child1, child2)
                    else:
                        toolbox.mate_two(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation to each individual
            for idx, indiv in enumerate(offspring):
                toolbox.mutate(indiv)
                del indiv.fitness.values

            # Update the fitness values            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population and update hall of fame
            pop[:] = offspring
            it_stats = stats.compile(pop)
            hof.update(pop)
            print('{}\t\t{}\t\t{}\t\t{}\t{}'.format(
                it, 
                np.around(it_stats['avg'], 4), 
                np.around(it_stats['std'], 3), 
                np.around(it_stats['max'], 6),
                np.around(time.time() - start, 4), 
            ))

            if it_stats['max'] > best_score:
                best_it = it
                best_score = it_stats['max']

            it += 1

        return hof


class ParticleSwarmExtractor(Extractor):
    def __init__(self, particles=50, iterations=25, verbose=True, wait=5,
                 smin=-0.25, smax=0.25, phi1=1, phi2=1):
        self.particles = particles
        self.iterations = iterations
        self.verbose = verbose
        self.wait = wait
        self.smin = smin
        self.smax = smax
        self.phi1 = phi1
        self.phi2 = phi2

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(ParticleSwarmExtractor, self).extract(timeseries, labels, min_len,
                                                    max_len, nr_shapelets, metric)

        def random_shapelet():
            rand_row_idx = np.random.randint(self.timeseries.shape[0])
            rand_length = np.random.choice(range(self.min_len, self.max_len), size=1)[0]
            rand_col_start_idx = np.random.randint(self.timeseries.shape[1] - rand_length)
            return self.timeseries[
                rand_row_idx, 
                rand_col_start_idx:rand_col_start_idx+rand_length
            ]

        def generate(smin, smax, n):
            parts = []
            for _ in range(n):
                rand_shap = random_shapelet()
                part = creator.Particle(rand_shap)
                part.speed = np.random.uniform(smin, smax, len(rand_shap))
                part.smin = smin
                part.smax = smax
                parts.append(part)
            return parts

        def updateParticle(part, best, phi1, phi2):
            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            #TODO: recheck this out (what if particles have variable lengths??)
            if len(part) < len(best):
                d, pos = util.sdist_with_pos(part, best)
                v_u1 = u1 * (part.best - part)
                v_u2 = u2 * (best[pos:pos+len(part)] - part)
                # These magic numbers are found in http://www.ijmlc.org/vol5/521-C016.pdf
                part.speed = 0.729*part.speed + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
                part += part.speed
            else:
                d, pos = util.sdist_with_pos(best, part)
                v_u1 = (u1 * (part.best - part))[pos:pos+len(best)]
                v_u2 = u2[pos:pos+len(best)] * (best - part[pos:pos+len(best)])
                # These magic numbers are found in http://www.ijmlc.org/vol5/521-C016.pdf
                part.speed[pos:pos+len(best)] = 0.729*part.speed[pos:pos+len(best)] + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
                part[pos:pos+len(best)] += part.speed[pos:pos+len(best)]


        def cost(shapelet):
            L = []
            for k in range(len(self.timeseries)):
                D = self.timeseries[k, :]
                dist = util.sdist(shapelet, D)
                L.append((dist, self.labels[k]))
            return self.metric(L)

        weights = (1.0,)
        if metric == 'ig':
            weights = (1.0, 1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, 
                       speed=list, smin=None, smax=None, best=None)

        toolbox = base.Toolbox()
        toolbox.register("population", generate, smin=self.smin, smax=self.smax)
        toolbox.register("update", updateParticle, phi1=self.phi1, phi2=self.phi2)
        toolbox.register("evaluate", cost)

        pop = toolbox.population(n=self.particles)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = 10000
        best = None
        it_wo_improvement = 0
        g = 0

        for g in range(GEN):
            it_wo_improvement += 1
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    print('--->', part.fitness)
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    it_wo_improvement = 0
            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

            if it_wo_improvement >= self.wait:
                break

        return [best]

class ParticleSwarmExtractor2(Extractor):
    # TODO: store a best particle per length
    # TODO: implement length-adjusting strategies
    def __init__(self, particles=50, iterations=25, verbose=True, wait=5,
                 smin=-0.25, smax=0.25, phi1=1, phi2=1, length_mut_prob=0.33):
        self.particles = particles
        self.iterations = iterations
        self.verbose = verbose
        self.wait = wait
        self.smin = smin
        self.smax = smax
        self.phi1 = phi1
        self.phi2 = phi2
        self.length_mut_prob = length_mut_prob

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(ParticleSwarmExtractor2, self).extract(timeseries, labels, min_len,
                                                    max_len, nr_shapelets, metric)

        def random_shapelet():
            rand_row_idx = np.random.randint(self.timeseries.shape[0])
            rand_length = np.random.choice(range(self.min_len, self.max_len), size=1)[0]
            rand_col_start_idx = np.random.randint(self.timeseries.shape[1] - rand_length)
            return self.timeseries[
                rand_row_idx, 
                rand_col_start_idx:rand_col_start_idx+rand_length
            ]

        def generate(smin, smax, n):
            parts = []
            for _ in range(n):
                rand_shap = random_shapelet()
                part = creator.Particle(rand_shap)
                part.speed = np.random.uniform(smin, smax, len(rand_shap))
                part.smin = smin
                part.smax = smax
                parts.append(part)
            return parts

        def update_values(part, new_values):
            new_part = creator.Particle(new_values)
            new_part.speed = part.speed
            new_part.smin = part.smin
            new_part.smax = part.smax
            new_part.best = part.best
            return new_part

        def updateParticle(part, global_best, best_per_length, phi1, phi2, length_mut_prob):
            if np.random.random() < length_mut_prob:
                diff = len(global_best) - len(part)
                if diff != 0: 
                    rand_nr_elements = np.random.randint(abs(diff))
                if diff > 0:
                    rand_elts = list(np.random.uniform(-1, 1, rand_nr_elements))
                    if np.random.random() < 0.5:
                        # Add elements at start
                        part = update_values(part, np.array(rand_elts + list(part)))
                        part.speed = np.array(list(np.random.uniform(part.smin, part.smax, rand_nr_elements)) + list(part.speed))
                        part.best = np.array(rand_elts + list(part.best))
                    else:
                        # Add elements at end
                        part = update_values(part, np.array(list(part) + rand_elts))
                        part.speed = np.array(list(part.speed) + list(np.random.uniform(part.smin, part.smax, rand_nr_elements)))
                        part.best = np.array(list(part.best) + rand_elts)
                elif diff < 0:
                    if np.random.random() < 0.5:
                        # Remove elements at start
                        part = update_values(part, part[rand_nr_elements:])
                        part.speed = part.speed[rand_nr_elements:]
                        part.best = part.best[rand_nr_elements:]
                    else:
                        # Remove elements at end
                        part = update_values(part, part[:-rand_nr_elements])
                        part.speed = part.speed[:-rand_nr_elements]
                        part.best = part.best[:-rand_nr_elements]

            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            if len(part) in best_per_length:
                length_best = best_per_length[len(part)]
                v_u1 = u1 * (part.best - part)
                v_u2 = u2 * (length_best - part)
                part.speed = 0.729*part.speed + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
                part += part.speed


        def cost(shapelet):
            L = []
            for k in range(len(self.timeseries)):
                D = self.timeseries[k, :]
                dist = util.sdist(shapelet, D)
                L.append((dist, self.labels[k]))
            return self.metric(L)

        weights = (1.0,)
        if metric == 'ig':
            weights = (1.0, 1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, 
                       speed=list, smin=None, smax=None, best=None)

        toolbox = base.Toolbox()
        toolbox.register("population", generate, smin=self.smin, smax=self.smax)
        toolbox.register("update", updateParticle, phi1=self.phi1, phi2=self.phi2, length_mut_prob=self.length_mut_prob)
        toolbox.register("evaluate", cost)

        pop = toolbox.population(n=self.particles)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = 10000
        best = None
        best_per_length = {}
        it_wo_improvement = 0
        g = 0

        for g in range(GEN):
            it_wo_improvement += 1
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    print('--->', part.fitness)
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    it_wo_improvement = 0
                if len(part) not in best_per_length or best_per_length[len(part)].fitness < part.fitness:
                    best_per_length[len(part)] = creator.Particle(part)
                    best_per_length[len(part)].fitness.values = part.fitness.values

            for part in pop:
                toolbox.update(part, best, best_per_length)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

            if it_wo_improvement == self.wait:
                break

        return [best]


class SAXExtractor(Extractor):
    def __init__(self, alphabet_size=4, sax_length=16, nr_candidates=100, 
                 iterations=5, mask_size=3):
        super(SAXExtractor, self).__init__()
        self.alphabet_size = alphabet_size
        self.sax_length = sax_length
        self.nr_candidates = nr_candidates
        self.iterations = iterations
        self.mask_size = mask_size

    def _random_mask(self, sax_timeseries, mask_size=5):
        """In order to calculate similarity between different timeseries
        in the SAX domain, we apply random masks and check whether the 
        remainder of the timeseries are equal to eachother.

        Parameters:
        -----------
        * sax_timeseries (3D np.array: timeseries x sax_words x word_length)
             The timeseries to mask
        * mask_size (int)
             How many elements should be masked
        """
        random_idx = np.random.choice(
            range(sax_timeseries.shape[2]),
            size=sax_timeseries.shape[2] - mask_size,
            replace=False
        )
        return sax_timeseries[:, :, random_idx]


    def _create_score_table(self, sax_timeseries, labels, iterations=10, 
                            mask_size=5):
        unique_labels = list(set(labels))
        score_table = np.zeros((
            sax_timeseries.shape[0], 
            sax_timeseries.shape[1],
            len(unique_labels)
        ))

        for it in range(iterations):
            masked_timeseries = self._random_mask(sax_timeseries, mask_size)
            hash_table = defaultdict(list)
            for ts_idx in range(masked_timeseries.shape[0]):
                for sax_idx in range(masked_timeseries.shape[1]):
                    key = tuple(list(masked_timeseries[ts_idx, sax_idx]))
                    hash_table[key].append((ts_idx, sax_idx))
            
            for bucket in hash_table:
                for (ts_idx1, sax_idx) in hash_table[bucket]:
                    unique_idx = set([x[0] for x in hash_table[bucket]])
                    for idx in unique_idx:
                        score_table[
                            ts_idx1, 
                            sax_idx, 
                            unique_labels.index(labels[idx])
                        ] += 1

        return score_table

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        super(SAXExtractor, self).extract(timeseries, labels, min_len,
                                          max_len, nr_shapelets, metric)

        if self.min_len == 1:
            self.min_len = self.sax_length

        unique_classes = set(self.labels)
        classes_cntr = Counter(self.labels)

        shapelets = []
        for l in trange(self.min_len, self.max_len, desc='length', position=0):
            # To select the candidates, all subsequences of length l from   
            # all time series are created using the sliding window technique, 
            # and we create their corresponding SAX word and keep them in SAXList 
            sax_words = np.zeros((
                len(self.timeseries), 
                self.timeseries.shape[1] - l + 1,
                self.sax_length
            ))
            for ts_idx, ts in enumerate(self.timeseries):
                # Extract all possible subseries, by using a sliding window
                # with shift=1
                subseries = []
                for k in range(len(ts) - l + 1):
                    subseries.append(util.z_norm(ts[k:k+l]))
                # Transform all the subseries and add them to the sax_words
                transformed_timeseries = transform(subseries, self.sax_length, 
                                                   self.alphabet_size)
                sax_words[ts_idx] = transformed_timeseries
            
            score_table = self._create_score_table(sax_words, self.labels, 
                                                   iterations=self.iterations,
                                                   mask_size=self.mask_size)
            max_score_table = np.ones_like(score_table)
            for c in unique_classes:
                max_score_table[:, :, c] = classes_cntr[c] * self.iterations
            rev_score_table = max_score_table - score_table

            # TODO: Can we replace this simple power calculation by a more
            # powerful metric to heuristically measure the quality
            power = []
            for ts_idx in range(score_table.shape[0]):
                for sax_idx in range(score_table.shape[1]):
                    min_val, max_val = float('inf'), float('-inf')
                    total = 0
                    for class_idx in range(score_table.shape[2]):
                        score = score_table[ts_idx, sax_idx, class_idx]
                        rev_score = rev_score_table[ts_idx, sax_idx, class_idx]
                        diff = score - rev_score
                        if diff > max_val:
                            max_val = diff
                        if diff < min_val:
                            min_val = diff
                        total += abs(diff)

                    v = (total-abs(max_val)-abs(min_val)) + abs(max_val-min_val)
                    power.append((v, (ts_idx, sax_idx)))
            
            top_candidates = sorted(power, key=lambda x: -x[0])[:self.nr_candidates]
            for score, (ts_idx, sax_idx) in top_candidates:
                candidate = self.timeseries[ts_idx][sax_idx:sax_idx+l]
                L = []  # The orderline, to calculate entropy
                for k in range(len(self.timeseries)):
                    D = self.timeseries[k, :]
                    dist = util.sdist(candidate, D)
                    L.append((dist, self.labels[k]))
                score = self.metric(L)
                shapelets.append(([list(candidate)] + list(score) + [ts_idx, sax_idx, l]))

        shapelets = sorted(shapelets, key=self.key)
        best_shapelets = extract_top_k_shapelets(shapelets, nr_shapelets, self.key)
        return best_shapelets