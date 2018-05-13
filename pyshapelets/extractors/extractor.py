# Standard library
from collections import defaultdict, Counter

# pip-installable libraries
import pandas as pd
import numpy as np
from tqdm import trange

# Some beautiful Python imports
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sax_transform import transform
import util

from tslearn.shapelets import ShapeletModel

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
                nr_shapelets=1):
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


class BruteForceExtractor(Extractor):
    def __init__(self):
        pass

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1):
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
                    L = sorted(L, key=lambda x: x[0])
                    tau, gain, gap = util.calculate_ig(L)
                    shapelets.append((candidate, tau, gain, gap))

        shapelets = sorted(shapelets, key=lambda x: (-x[2], -x[3]))
        best_shapelets = [(x[0], x[1]) for x in shapelets[:nr_shapelets]]
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
                nr_shapelets=1):
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
                    L = sorted(L, key=lambda x: x[0])
                    tau, gain, gap = util.calculate_ig(L)
                    shapelets.append((S[i:i+l], tau, gain, gap))

                    if self.pruning:
                        H.put((L, S[i:i+l]))

        shapelets = sorted(shapelets, key=lambda x: (-x[2], -x[3]))
        best_shapelets = [(x[0], x[1]) for x in shapelets[:nr_shapelets]]
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
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1):
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
            L = sorted(L, key=lambda x: x[0])
            tau, gain, gap = util.calculate_ig(L)
            shapelets.append((candidate, tau, gain, gap))

        shapelets = sorted(shapelets, key=lambda x: (-x[2], -x[3]))
        best_shapelets = [(x[0], x[1]) for x in shapelets[:nr_shapelets]]
        return best_shapelets


class GeneticExtractor(Extractor):
    pass


class ParticleSwarmExtractor(Extractor):
    pass


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

        #TODO: If stuff ever bugs out, check here first...
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
                nr_shapelets=1):
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
                L = sorted(L, key=lambda x: x[0])
                tau, gain, gap = util.calculate_ig(L)
                shapelets.append((candidate, tau, gain, gap))

        shapelets = sorted(shapelets, key=lambda x: (-x[2], -x[3]))
        best_shapelets = [(x[0], x[1]) for x in shapelets[:nr_shapelets]]
        return best_shapelets