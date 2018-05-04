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

class Extractor(object):
    def __init__(self, min_len=None, max_len=None, nr_shapelets=1):
        # These hyper-parameters are shared over all extractors
        self.nr_shapelets = nr_shapelets
        self.min_len = min_len
        self.max_len = max_len

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

    def extract(self, timeseries, labels):
        self.timeseries, self.labels = self._convert_to_numpy(timeseries, 
                                                              labels)

        # If no min_len and max_len are provided, we initialise them
        if self.min_len is None:
            self.min_len = 1
        if self.max_len is None:
            self.max_len = self.timeseries.shape[1]


class SAXExtractor(Extractor):
    def __init__(self, min_len=None, max_len=None, nr_shapelets=1, 
                 alphabet_size=4, sax_length=8, nr_candidates=25, 
                 iterations=5, mask_size=3):
        super(SAXExtractor, self).__init__(min_len=min_len,
                                           max_len=max_len, 
                                           nr_shapelets=nr_shapelets)
        self.alphabet_size = alphabet_size
        self.sax_length = sax_length
        self.nr_candidates = nr_candidates
        self.iterations = iterations
        self.mask_size = mask_size

    def _random_mask(self, sax_timeseries, mask_size=5):
        """When discretizing a continous real-valued timeseries, the problem of
        false dismissals arises. This is caused by the fact that two timeseries that
        differ only by a tiny epsilon can result in two different words. To 
        alleviate this problem, we take random subsets of each word by masking
        them. Now a trade-off between false dismissals and false positives must
        be considered. The higher the mask_size, the higher the probability of
        false positives.
        Parameters:
        -----------
         * sax_timeseries (3D np.array: timeseries x sax_words x word_length)
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

    def extract(self, timeseries, labels):
        super(SAXExtractor, self).extract(timeseries, labels)

        if self.min_len == 1:
            self.min_len = self.sax_length

        unique_classes = set(self.labels)
        classes_cntr = Counter(self.labels)

        max_gain, max_gap = 0, 0
        best_shapelet, best_dist, best_L = None, None, None
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
                tau, updated, new_gain, new_gap = util.best_ig(L, max_gain, 
                                                               max_gap)
                if updated:
                    best_shapelet = candidate
                    print('Found new best shapelet of length {} with gain {} and gap {}'.format(len(best_shapelet), new_gain, new_gap))
                    best_dist = tau
                    best_L = L
                    max_gain = new_gain
                    max_gap = new_gap

        return best_shapelet, best_dist, best_L, max_gain, max_gap


extractor = SAXExtractor()
extractor.extract([[0,0,0,0], [1,1,1,1]], [0, 1])