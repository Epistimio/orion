# Roadmap
Last update February 12th, 2020

## Next releases - Short-Term

### v0.2: ETA End of summer 2019
#### Journal Protocol Plugins
Offering:
- no need to setup DB, can use one's existing backend
- Can re-use tools provided by backend for visualizations, etc.
#### Python API

Traditional `suggest`/`observe` interface

```python
experiment = orion.client.register(
    experiment='fct-dummy',
    x='loguniform(0.1, 1)', verbose=True, message='running trial {trial.hash_name}')

trial = experiment.suggest()
results = dummy(**trial.arguments)
experiment.observe(trial, results)
```

## Next releases - Mid-Term

### v0.3
#### Generic `Optimizer` interface supporting various types of algorithms

Change interface to support trial object instead of curated lists. This is necessary to support algorithms such as PBT.

#### More Optimizers
- PBT
- BOHB

## Next releases - Long-Term

#### Simple dashboard specific to monitoring and benchmarking of Black-Box optimization
- Specific to hyper parameter optimizations
- Provide status of experiments

#### Conditional Space

The Space class will be refactored on top of [ConfigSpace](https://automl.github.io/ConfigSpace). This will give access to [conditional parameters](https://automl.github.io/ConfigSpace/master/Guide.html#nd-example-categorical-hyperparameters-and-conditions) and [forbidden clauses](https://automl.github.io/ConfigSpace/master/Guide.html#rd-example-forbidden-clauses).

## Other possibilities
- NAS
- Multi-Objective Optimization
- Dynamic Scheduler
