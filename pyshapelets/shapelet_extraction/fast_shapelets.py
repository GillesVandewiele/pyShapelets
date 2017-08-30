from collections import Counter
import time
import numpy as np
import copy

from pyshapelets.shapelet_extraction.brute_force import find_best_split_point, check_candidate
from pyshapelets.util import util

import subprocess
import os

from pyshapelets.visualization.shapelet_tree import ShapeletTree


# def check_candidate(timeseries, labels, shapelet, stats):
#     distances = []
#     for index, time_serie, label in enumerate(zip(timeseries, labels)):
#         d, idx = util.subsequence_dist_z_space(time_serie, shapelet, stats[index])
#         distances.append((d, label))
#
#     gain, dist = find_best_split_point(sorted(distances, key=lambda x: x[0]))
#     distances_left = []
#     distances_right = []
#     for index, time_serie in enumerate(timeseries):
#         d = util.subsequence_dist_z_space(time_serie, shapelet, stats[index])
#         if d <= dist:
#             distances_left.append(d)
#         else:
#             distances_right.append(d)
#
#     return gain, dist, np.mean(distances_right) - np.mean(distances_left)


def C_wrapper(timeseries, labels, exe_location='../shapelet/shapelet_best.exe'):
    min_len = min([len(x) for x in timeseries])

    # The executable works with classes from 0 to #classes. Make a (reverse) mapping
    class_mapping = {}
    class_mapping_rev = {}
    cntr = 0
    for _class in np.unique(labels):
        class_mapping[_class] = cntr
        class_mapping_rev[cntr] = _class
        cntr += 1

    labels = np.vectorize(class_mapping.get)(labels)

    # Create a file Temp_TRAIN, acceptable for the executable
    lines = []
    for ts, label in zip(timeseries, labels):
        # First digit on the line is the label, the next digits are the time serie. Sep between digits is a tab
        lines.append(np.array([label] + list(ts)))

    lines = np.array(lines)
    np.savetxt('Temp_TRAIN', lines)

    # Call the executable
    subprocess.call(['./'+exe_location, 'Temp_TRAIN', str(len(timeseries)), str(len(np.unique(labels))),
                     str(min_len), '1', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # The executable will write a new file called Temp_TRAIN_tree, parse it!
    with open('Temp_TRAIN_tree', 'r') as fp:
        lines = fp.readlines()
        i = 0
        nodes = {}  # Store the parsed nodes in this dict (key = node_id)
        while i < len(lines):
            if not lines[i+1].startswith('0'):
                # Read in a block of 4 lines (internal node)
                id = int(lines[i].rstrip())
                distance = float(lines[i+2].rstrip())
                shapelet = list(map(float, lines[i+3].rstrip().split()))
                # max_ig, best_distance = check_candidate(timeseries, labels, shapelet)
                # print('------>', best_distance, distance)
                nodes[id] = ShapeletTree(distance=distance, shapelet=shapelet)

                i += 4
            else:
                # Block of 3 lines (leaf)
                id = int(lines[i].rstrip())
                _class = class_mapping_rev[int(lines[i+2].rstrip())]
                class_probabilities = {}
                for key in class_mapping.keys(): class_probabilities[key] = [0, 1][key == _class]

                nodes[id] = ShapeletTree(class_probabilities=class_probabilities)

                i += 3

    # Construct the tree (recursively divide the node by two to determine path from root)
    #                           1
    #                   2               3
    #               4       5       6       7
    root = nodes[1]
    for node in sorted(nodes.keys()):
        if node > 1:
            k = node
            directions = []
            while k > 1:
                directions.append(k % 2)
                k //= 2

            parent = root
            for direction in directions[::-1][:-1]:
                parent = [parent.left, parent.right][direction]

            if directions[0]:
                parent.right = nodes[node]
            else:
                parent.left = nodes[node]

    # Delete Temp_TRAIN and Temp_TRAIN_tree
    # os.remove('Temp_TRAIN')
    # os.remove('Temp_TRAIN_tree')

    labels = np.vectorize(class_mapping_rev.get)(labels)

    # root.recalculate_distances(timeseries, labels)

    return root


def fast_shapelet_discovery(timeseries, labels, m=None):
    if m is None:
        m = np.min([len(x) for x in timeseries]) # Maximum length of a timeserie
    max_gain, max_gap = 0, 0
    best_shapelet, best_distance, best_L = None, None, None
    cntr = 0
    for ts, label in zip(timeseries, labels):
        print(cntr, '/', len(timeseries))
        cntr += 1

        x = (ts, label)
        stats = {}
        for i, (ts2, label2) in enumerate(zip(timeseries, labels)):
            stats[i] = util.calculate_stats(ts, ts2)

        for l in range(1, m+1): # Possible shapelet lengths
            H = []  # Cache/history
            for i in range(len(ts) - l): # Possible start positions
                broken = False
                for (L, S) in H:
                    R = util.sdist(ts[i:i+l], S)
                    if util.upperIG(L, R, timeseries, labels) < max_gain:
                        broken = True
                        break # Continue with next i

                if not broken:
                    L = []
                    for k, (ts2, label2) in enumerate(zip(timeseries, labels)):
                        L.append((util.sdist_new(ts[i:i+l], ts2, i, stats[k]), label2))
                    L = sorted(L, key=lambda x: x[0])
                    print(L)
                    best_ig, tau = find_best_split_point(L)
                    if best_ig < max_gain:
                        best_shapelet = ts[i:i+l]
                        max_gain = best_ig
                        best_L = L
                        best_distance = tau
                        # (1.4578175811448, 1), (nan, 0), (nan, 1), (nan, 0), (nan, 1), ...
                        print('---->', max_gain, best_distance)
                    H.append((L, ts[i:i+l]))

    return best_shapelet, best_distance, best_L, max_gain

