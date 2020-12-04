# Roadmap
Last update December 3rd, 2020

## Next releases - Short-Term

### v0.1.12

#### Quick release for bug fixes

### v0.1.13

#### Default database backend
PickledDB will become the default database backend instead of MongoDB. This initiative reduces
initial configuration, allowing users to get started faster.

### v0.2
#### Journal Protocol Plugins
Offering:
- No need to setup DB, can use one's existing backend
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

### Algorithms
Introducing new algorithms: [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), [HyperBand](https://arxiv.org/abs/1603.06560)

## Next releases - Mid-Term

### v0.3
#### Generic `Optimizer` interface supporting various types of algorithms

Change interface to support trial object instead of curated lists. This is necessary to support algorithms such as PBT.

#### More Optimizers
- [PBT](https://arxiv.org/abs/1711.09846)
- [BOHB](https://ml.informatik.uni-freiburg.de/papers/18-ICML-BOHB.pdf)

#### Simple dashboard specific to monitoring and benchmarking of Black-Box optimization
- Specific to hyper parameter optimizations
- Provide status of experiments

#### Leveraging previous experiences
Leveraging the knowledge base contained in the EVC of previous trials to optimize and drive new
 trials.

## Next releases - Long-Term

#### Conditional Space

The Space class will be refactored on top of [ConfigSpace](https://automl.github.io/ConfigSpace). This will give access to [conditional parameters](https://automl.github.io/ConfigSpace/master/Guide.html#nd-example-categorical-hyperparameters-and-conditions) and [forbidden clauses](https://automl.github.io/ConfigSpace/master/Guide.html#rd-example-forbidden-clauses).

## Other possibilities
- NAS
- Multi-Objective Optimization
- Dynamic Scheduler
