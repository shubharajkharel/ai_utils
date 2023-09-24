# Optuna

## Optuna workflow

- Define an objective function `Study` is an optimization session, which is a set of trials
  - Specify the optimization `direction`, `sampler`, `pruner`, etc.
  - `optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)`
- `Trial` is a single execution of the objective function
- `Study.optimize` method starts the optimization:
- Save the study object for later use and visualization
- Find the best trial, best parameters, etc.:
  - `study.best_trial`, `study.best_params`, etc.

## Recommended Samplers and Pruner

- sampler [list](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)
- For (**NON-ML tasks**)[https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako]:
  - For `RandomSampler`, `MedianPruner` is the best.
  - For `TPESampler`, `HyperbandPruner` is the best.
- For (**ML tasks**)[https://search.ieice.org/bin/summary.php?id=j103-d_9_615&category=-D&year=2020&lang=J&abst=]:
  - (See)[https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning:~:text=written%20in%20Japanese.-,Parallel,-Compute%20Resource]

## ML Workflow Objective

- **ML Objective**:

  - Define a model, dataset, trainer, etc.
  - Speicify pruner and sampler when creating the study
  - use `trial.suggest_*` to specify hyperparameters:
    - Specify `search space` for suggestions
      - Pythonic way:
        - `trial.suggest_int("num_layers", 1, 3)`
        - `trial.suggest_float("dropout_rate", 0.0, 1.0)`
        - `trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])`
  - train the model
  - return the optimization metric like validation accuracy, loss, FLOPS, etc.
    - `rainer.callback_metrics["val_acc"].item()`

- **ML Pruning**:

  - Easy integration with ML frameworks like Pytorch, Pytorch Lighning and more:
  - Generally:
    - raise optuna `TrialPruned` exception when the trial is unpromising
    - Report must be generated before checking if the trial should be
      - eg: `trial.report(val_acc, epoch)`
    - Raise Optuna Exception to handle pruning
      - `if trial.should_prune(): raise optuna.TrialPruned()`
  - Easy integration with Pytorch Lighning:
    - `pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_acc")`
    - eg: `trainer = pl.Trainer(.., callbacks=[pruning_callback, ..])`
  - Examples: [Pytorch Lightning](https://github.com/optuna/optuna-examples/tree/main/pytorch)

- **Useful Metrics for ML**:
  - FLOP Count from (Facebook Research)[https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md]

## Distributed Optimization

- Use the frameworks parallelism: eg: `optuna.integration.TorchDistributedTrial`

## Visualizations

- **Optuna Dashboard**:

  - Optuna Dashboard `optuna-dashboard sqlite:///db.sqlite3`
  - (Browser only version)[https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html#browser-only-version-experimental] and (VS Code extension)[https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html#vs-code-extension-experimental] seems to show fewer plots/info

- Visualize the optimization history, hyperparameter importances, etc. using
  using functions in `optuna.visualization` module
  - `plot_contour` : parameter relationship as contour plot in a study.
  - `plot_edf` : objective value EDF (empirical distribution function)
  - `plot_hypervolume_history` : hypervolume history of all trials in a study.
  - `plot_intermediate_values` : intermediate values of all trials in a study.
  - `plot_optimization_history` :optimization history of all trials in a study.
  - `plot_parallel_coordinate` : high-dimensional parameter relationships
  - `plot_param_importances` : hyperparameter importances.
  - `plot_pareto_front` : the Pareto front of a study.
  - `plot_rank` :
    - parameter relations as scatter plots
    - with colors indicating ranks of target value.
  - `plot_slice` : parameter relationship as slice plot in a study.
  - `plot_terminator_improvement` : potentials for future improvement.
  - `plot_timeline` : timeline of a study.
