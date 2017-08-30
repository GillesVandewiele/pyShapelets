from evostra import EvolutionStrategy
from pyshapelets.shapelet_extraction.brute_force import generate_candidates, check_candidate

import numpy as np

def find_shapelets_es(timeseries, labels, max_len=100, min_len=1, population_size=100,
                       iterations=25, verbose=True, sigma=0.1, learning_rate=0.001):

    def cost(shapelet):
        return check_candidate(timeseries, labels, shapelet)[0]

    candidates = np.array(generate_candidates(timeseries, labels, max_len, min_len))

    es = EvolutionStrategy(candidates[np.random.choice(range(len(candidates)), size=population_size)][0][0],
                           cost, population_size=population_size, sigma=sigma, learning_rate=learning_rate)
    es.run(iterations, print_step=1)

    best_shapelet = es.get_weights()
    return best_shapelet