import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sax_transform import transform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
from collections import defaultdict, Counter
from tqdm import trange

def random_mask(sax_timeseries, mask_size=5):
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


def create_score_table(sax_timeseries, labels, iterations=10, mask_size=5):
    unique_labels = list(set(labels))
    score_table = np.zeros((
        sax_timeseries.shape[0], 
        sax_timeseries.shape[1],
        len(unique_labels)
    ))

    #TODO: If stuff ever bugs out, check here first...
    for it in range(iterations):
        masked_timeseries = random_mask(sax_timeseries, mask_size)
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


def extract_shapelet(timeseries, labels, alphabet_size=4, sax_length=8, 
                     nr_candidates=25, iterations=5, mask_size=3, min_len=None, 
                     max_len=None):
    # If no min_len and max_len are provided, we fill then in ourselves
    if min_len is None or min_len < sax_length:
        min_len = sax_length
    if max_len is None:
        max_len = timeseries.shape[1]

    if type(timeseries) == pd.DataFrame or type(timeseries) == pd.Series:
        timeseries = timeseries.values
    if type(labels) == pd.DataFrame or type(labels) == pd.Series:
        labels = labels.values

    unique_classes = set(labels)
    classes_cntr = Counter(labels)

    max_gain, max_gap = 0, 0
    best_shapelet, best_dist, best_L = None, None, None
    for l in trange(min_len, max_len, desc='length', position=0):
        # To select the candidates, all subsequences of length l from   
        # all time series are created using the sliding window technique, 
        # and we create their corresponding SAX word and keep them in SAXList 
        sax_words = np.zeros((
            len(timeseries), 
            timeseries.shape[1] - l + 1,
            sax_length
        ))
        for ts_idx, ts in enumerate(timeseries):
            # Extract all possible subseries, by using a sliding window
            # with shift=1
            subseries = []
            for k in range(len(ts) - l + 1):
                subseries.append(util.z_norm(ts[k:k+l]))
            # Transform all the subseries and add them to the sax_words
            transformed_timeseries = transform(subseries, sax_length, 
                                               alphabet_size)
            sax_words[ts_idx] = transformed_timeseries
        
        score_table = create_score_table(sax_words, labels, 
                                         iterations=iterations,
                                         mask_size=mask_size)
        max_score_table = np.ones_like(score_table)
        for c in unique_classes:
            max_score_table[:, :, c] = classes_cntr[c] * iterations
        rev_score_table = max_score_table - score_table

        power = []
        for ts_idx in range(score_table.shape[0]):
            for sax_idx in range(score_table.shape[1]):
                min_val, max_val = float('inf'), float('-inf')
                total = 0
                for class_idx in range(score_table.shape[2]):
                    diff = score_table[ts_idx, sax_idx, class_idx] - rev_score_table[ts_idx, sax_idx, class_idx]
                    if diff > max_val:
                        max_val = diff
                    if diff < min_val:
                        min_val = diff
                    total += abs(diff)

                v = (total-abs(max_val)-abs(min_val)) + abs(max_val-min_val)
                power.append((v, (ts_idx, sax_idx)))
        
        top_candidates = sorted(power, key=lambda x: -x[0])[:nr_candidates]
        for score, (ts_idx, sax_idx) in top_candidates:
            candidate = timeseries[ts_idx][sax_idx:sax_idx+l]
            L = []  # The orderline, to calculate entropy
            for k in range(len(timeseries)):
                D = timeseries[k, :]
                dist = util.sdist(candidate, D)
                L.append((dist, labels[k]))
            L = sorted(L, key=lambda x: x[0])
            tau, updated, new_gain, new_gap = util.best_ig(L, max_gain, max_gap)
            if updated:
                best_shapelet = candidate
                print('Found new best shapelet of length {} with gain {} and gap {}'.format(len(best_shapelet), new_gain, new_gap))
                best_dist = tau
                best_L = L
                max_gain = new_gain
                max_gap = new_gap

    return best_shapelet, best_dist, best_L, max_gain, max_gap


    