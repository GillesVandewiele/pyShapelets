# pyShapelets
A python library that implements different algorithms to extract shapelets from timeseries.

Dependencies are quite minimal:

  * NumPy
  * SciPy
  * DEAP (for the bio-inspired algorithms)

The algorithms implemented are the following:

  * **Brute Force**: with early search abandon and entropy pre-pruning
  [Time series shapelets: a new primitive for data mining](http://dl.acm.org/citation.cfm?id=1557122)
  
  * **Genetic algorithm** (could not really find a paper)
  
  * **Particle Swarm Optimization**
  [Time Series Shapelets: Training Time Improvement Based on Particle Swarm Optimization](http://www.ijmlc.org/vol5/521-C016.pdf)


The following algorithms are on my TODO:

  * **Dynamic programming logical-shapelets**
  [Logical-Shapelets: An Expressive Primitive for Time Series Classification](http://www.cs.ucr.edu/~eamonn/LogicalShapelet.pdf)
  
  * **Particle Swarm inspired Evolutionary Algorithm (PS-EA)**
  [Particle swarm inspired evolutionary algorithm (PS-EA) for multiobjective optimization problems](http://ieeexplore.ieee.org/abstract/document/1299374/)
