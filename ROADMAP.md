# Roadmap
Last update March 7th, 2022

## Next releases - Short-Term

### v0.2.4

- [DEBH](https://arxiv.org/abs/2105.09821)
- [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO/archived_submissions/hebo)
- [BOHB](https://ml.informatik.uni-freiburg.de/papers/18-ICML-BOHB.pdf)
- [Nevergrad](https://github.com/facebookresearch/nevergrad)
- [Ax](https://ax.dev/)
- [MOFA](https://github.com/Epistimio/orion.algo.mofa)
- [PB2](https://github.com/Epistimio/orion.algo.pb2)
- Integration with Hydra
- Integration with [sample-space](https://github.com/Epistimio/sample-space) and
  [ConfigSpace](https://automl.github.io/ConfigSpace/master/)

## Next releases - Mid-Term

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
