# Roadmap
Last update Nov 23rd, 2021

## Next releases - Short-Term

### v0.2.1

- New master process to enhance parallelisation efficiency.
- [PBT](https://arxiv.org/abs/1711.09846)

### v0.2.2

- Use shared algo serialization instead of replications to enhance parallelisation efficiency.
- [DEBH](https://arxiv.org/abs/2105.09821)

### v0.2.3

- [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO/archived_submissions/hebo)

### v0.2.4

- [BOHB](https://ml.informatik.uni-freiburg.de/papers/18-ICML-BOHB.pdf)

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
