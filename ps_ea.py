from brute_force import generate_candidates, check_candidate

import numpy as np
from deap import base, creator, algorithms, tools
import matplotlib.pyplot as plt


def find_shapelets_ps_ea(timeseries, labels, max_len=100, min_len=1, particles=25,
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

    def updateParticle(part, gbest, pbest, popbest, neighbors, probs):
        for j in range(len(part)):
            part[j] = np.random.choice([gbest[j], pbest[j], popbest[j], part[j],
                                        part[j] + np.random.random()], size=1, p=probs)[0]
        return part

    def adjustProbabilities(probs):
        for i in range(len(probs)):
            j = np.random.choice(len(probs), size=1)[0]
            p, p2 = probs[i], probs[j]
            mut = np.random.uniform(-min(p, p2), min(p, p2), size=1)[0]
            probs[i] += mut
            probs[j] -= mut
        return probs

    def cost(shapelet):
        return (check_candidate(timeseries, labels, shapelet)[0],)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, smin=-0.25, smax=0.25)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle)
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
    probs = [0.1,0.1,0.1,0.5,0.2]

    prev_fitness = 0
    for g in range(GEN):
        print(probs)
        popbest = None
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
            if popbest is None or part.fitness > popbest.fitness:
                popbest = part
        for part in pop:
            toolbox.update(part, best, part.best, popbest, None, probs)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))

        # If we are not converging, then adjust our probabilities
        new_fitness = logbook.select(g, 'max')
        if new_fitness[1][0] - prev_fitness <= 0:
            probs = adjustProbabilities(probs)
        prev_fitness = new_fitness[1][0]
        print(logbook.stream)

    return pop, logbook, best