# Standard library
from collections import defaultdict, Counter
import array
import time

# pip-installable libraries
import pandas as pd
import numpy as np
from tqdm import trange
from deap import base, creator, algorithms, tools
from tslearn.shapelets import ShapeletModel

# Some beautiful Python imports
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sax_transform import transform
import util



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
                                                 max_len, nr_shapelets)
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
                    shapelets.append(([list(candidate)] + list(score)))

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
                                           max_len, nr_shapelets)
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
                    shapelets.append(([list(S[i:i+l])] + list(score)))

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
                                               max_len, nr_shapelets)

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


class GeneticExtractor(Extractor):
    def __init__(self, population_size=100, iterations=25, verbose=True,
                 mutation_prob=0.25, crossover_prob=0.25, wait=10):
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.wait = wait


    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric='ig'):
        # TODO: If nr_shapelets > 1, then represent individuals by
        # TODO: `nr_shapelets` shapelets (instead of taking top-k from hof)
        super(GeneticExtractor, self).extract(timeseries, labels, min_len,
                                              max_len, nr_shapelets)

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

        toolbox = base.Toolbox()
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)

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
            start = time.time()

            # Apply selection and cross-over the selected individuals
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation to each individual
            for indiv in offspring:
                if np.random.random() < self.mutation_prob:
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
    def __init__(self, particles=100, iterations=25, verbose=True, wait=10,
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
                                                    max_len, nr_shapelets)

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

        for g in range(GEN):
            it_wo_improvement += 1
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    it_wo_improvement = 0
            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

            if it_wo_improvement == self.wait:
                break

        return [best]


class SAXExtractor(Extractor):
    def __init__(self, alphabet_size=4, sax_length=8, nr_candidates=25, 
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
                                          max_len, nr_shapelets)

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
                shapelets.append(([list(candidate)]+list(score)))

        shapelets = sorted(shapelets, key=self.key)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets