# pyShapelets
A python library that implements different algorithms to extract shapelets from timeseries.

Dependencies are quite minimal:

  * NumPy
  * SciPy
  * DEAP (for the bio-inspired algorithms)

The algorithms implemented are the following:

  * Brute Force: with early search abandon and entropy pre-pruning
  [http://dl.acm.org/citation.cfm?id=1557122](Time series shapelets: a new primitive for data mining)
  
  * Genetic algorithm (could not really find a paper)
  
  * Particle Swarm Optimization
  [http://www.ijmlc.org/vol5/521-C016.pdf](Time Series Shapelets: Training Time Improvement Based on Particle Swarm Optimization)


The following algorithms are on my TODO:

  * Dynamic programming logical-shapelets
  [http://www.cs.ucr.edu/~eamonn/LogicalShapelet.pdf](Logical-Shapelets: An Expressive Primitive for Time Series Classification)
  
  * Particle Swarm inspired Evolutionary Algorithm (PS-EA)
  [http://ieeexplore.ieee.org/abstract/document/1299374/](Particle swarm inspired evolutionary algorithm (PS-EA) for multiobjective optimization problems)
