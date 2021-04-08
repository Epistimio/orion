# Roadmap
Last update December 3rd, 2020

## Next releases - Short-Term

### v0.1.15

#### Quick release for bug fixes

### v0.2
#### Native Multi-Processing support

Added support for parallelism with auto-scaling. No more need to launch multiple workers 
(though still supported) for parallelism, simply pass `--n-workers` or `workon(n_workers)`.

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
