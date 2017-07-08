from brute_force import generate_candidates, check_candidate

import numpy as np
from deap import base, creator, algorithms, tools
import matplotlib.pyplot as plt


def find_shapelets_pso(timeseries, labels, max_len=100, min_len=1, particles=25,
                       iterations=25, verbose=True):

    candidates = np.array(generate_candidates(timeseries, labels, max_len, min_len))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

    def generate(smin, smax):
        rand_cand = np.array(candidates[np.random.choice(range(len(candidates)), size=1)][0][0])
        part = creator.Particle(rand_cand)
        part.speed = np.random.uniform(smin, smax, len(rand_cand))
        part.smin = smin
        part.smax = smax
        return part

    def updateParticle(part, best, phi1, phi2):
        u1 = np.random.uniform(0, phi1, len(part))
        u2 = np.random.uniform(0, phi2, len(part))
        v_u1 = u1 * (part.best - part)
        v_u2 = u2 * (best - part)
        # These magic numbers are found in http://www.ijmlc.org/vol5/521-C016.pdf
        part.speed = 0.729*part.speed + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
        part += part.speed

    def cost(shapelet):
        return (check_candidate(timeseries, labels, shapelet)[0], )

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, smin=-0.25, smax=0.25)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=1, phi2=1)
    toolbox.register("evaluate", cost)

    pop = toolbox.population(n=particles)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best