# Multi-Objective Optimization

## Pareto Front

- A solution A `Dominate` B if it statisfies both:
  - A is better than B in at least one objective
  - A is no worse than B in all objectives
- `Optimal Solution(s)`: in multi-objective functions should not be dominated
  by any other solutions.
- The set of optimal solutions is called the `Pareto front`.
- Pareto Front contains non-dominated solutions i.e. one cannot improve a
  objective in the Pareto front without degrading another objective.
- ```text

     **Solutions**:
      A: Cost=5, Quality=8
      B: Cost=7, Quality=9
      C: Cost=6, Quality=7
      D: Cost=8, Quality=6

      **Domination**:
      A dominates C and D
      B dominates D
      C and D don't dominate anyone
  ```

## Optimality of Pareto front

- Quanification would be useful to compare different Pareto fronts, optimization algorithms and stopping criteria
- If pareto front can be `derived analytically` then we already have optimal
- In most cases, we do not know the analytical solution. To measure one, optimatility we need a `reference point` in the objective space.
- After optimization, the solutions in pareto front are compared to the reference point.
- The reference point is usually the `worst solution` in the objective space.
- **Hypervolume**:
  - Draw hypervolume (polygon) using reference point and pareto front.
  - Hypervolume for two and three objective optimization are area and volume respectively.
  - The shape of the hyper volume would be polygon for two objectives and polyhedron for three objectives.
  - See this [example](https://www.youtube.com/watch?v=cR4r1aNPBkQ) for visualization
  - For comparision between two alogirthms for example requires a reference point to be the same for both algorithms.
  - For same algorithm, the reference point can be the worst solution found by the algorithm.
  - As the search progresses, the hypervolume increases and can be used as a stopping criteria.

## Optmization Algorithms

- [Slides](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf)

## Multi-Objective Optimization with Optuna

- Directions must be specified for each objective:
  - `study = optuna.create_study(directions=["minimize", "maximize"])`
- Visualization:
  - `optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])`
