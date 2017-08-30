from pyshapelets.shapelet_extraction.brute_force import generate_candidates, check_candidate

import numpy as np
from deap import base, creator, algorithms, tools
import matplotlib.pyplot as plt


def find_shapelets_gen(timeseries, labels, max_len=100, min_len=1, population_size=100,
                       iterations=25, verbose=True):
    # Generate random subset of candidates, these represent our initial population
    candidates = np.array(generate_candidates(timeseries, labels, max_len, min_len))

    def get_random_candidate():
        return candidates[np.random.choice(range(len(candidates)), size=1)][0][0]

    def cost(shapelet):
        return (check_candidate(timeseries, labels, shapelet)[0], )

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.25)

    toolbox.register("individual", tools.initIterate, creator.Individual, get_random_candidate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", cost)
    toolbox.register("select", tools.selTournament, tournsize=10)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    pop = toolbox.population(n=population_size)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.25, ngen=100, stats=stats, verbose=True)

    gen = logbook.select("gen")
    fit_mins = logbook.select("max")
    size_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Maximum Fitness")
    line2 = ax1.plot(gen, size_avgs, "r-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower right", frameon=True)

    plt.show()