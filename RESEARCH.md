Intial Research Plan
====

The plan of this package is to build an automatic tool to do hyperparameter optimization. During our time at MILA, before and after coding up research ideas we always do hyperparameter optimization to set up a baseline on SOTA.
Previous methods which are grid search and random search requires tedious work on grad student part to handpicking parameters. We realize that this process could be automated.

Another point is that we want a tool that does not necessarily work for big labs like MILA, but for labs with just 5 computers with GPUs together connected with a router or a startup company in Silicon Valley that could only
afford low amount of AWS machines/GPUs, that doesn't have one reliable central node to control node coordination, new hyperparameter choice and node pruning. We plan to use a decentralize system that use effective broadcast of
hyperparameter choice to each node, and also saving the hyperparameter history on a TinyDB/levelDB. Although we would not hash all of the hyperparameter blocks to ensure we didn't choose the same hyperparameters across two different
nodes in the network.

The choice of new hyperparameters would be done using Bayesian optimization algorithms or evolutionary strategies/genetic algorithms. We would want to implement early stopping conditions such as Freeze-Thaw.

Execution of Research Plan
===

We plan to build this package that would be language agnostic, and also operation system agnostic, although at the beginning we would want to use it for mainly python scripts. This package is written mainly in C++17 and we would want to accomodate C++20. There would be a Python library written that you could call inside your script like `import metaopt` that would do the logging, and sending training statistics criteria to the database to be used to generate the new set of hyperparameters and also to early stop the experiment.

Summary of 04/10/2017 discussion and other notes
===

We would like our design to be aimed at satisfying the following 3 general requirements with descending priority order:

1. being **framework agnostic**:
    The training procedure is completely specified in its own executable, which will be executed by a child process (?).
2. being **language agnostic**:
    Same reason as (1). We will begin implementing an API for *Python* due to popularity. A client/worker side API will be designed to enable communication with the database and the server/master.
3. being **database agnostic**:
    This goal is to be able to run this software in a variety of web services/platforms. Maybe we can find a *Python* OSS, which wraps many popular databases, to act as a facade. We agreed that the initial implementation will be made in *MongoDB*.

Clarification on what is meant by *complete specification* of a training procedure
---
 * It uses some ML framework (we shouldn't care actually if it does).
 * It fetches the training dataset and the evaluation dataset.
 * Defines a model with its *parameter space* and part of architectural *hparam space*.
 * Defines a training optimization procedure with the rest of *hparam space*.
 * It will use our API to declare which *hparams* it should look for and which training *stats* it would export to the database.

So, the general goal, if I am thinking this right, is to decouple the training procedure itself from the hyperoptimization procedure.

**Worker-side** responsibilities
---
 * The *hparams* should be imported through our interface, using a YAML, json or another type of file which is to be set by the hyperoptimization procedure, and they will be also recorded inside the database as metadata of a particular training run.
 * Training *stats* are to be pushed by calls to our API within a worker's code, which can also trigger hooks to some master's procedure.
 * Part of our software library, within the worker process, will internally perform some of the calculations required by the hyperoptimization process, because:
   1. some data structures will be available in the workers only, e.g. evaluation sets for the hyperoptimization objective.
   2. concurrency
* Internal to our library, appropriate or desired modules will be loaded for the hyperoptimization procedure:
   1. A *hyperoptimization objective function*, which could be performance (accuracy, etc) on an hyperopt evaluation dataset or some form of regret function or anything which possibly takes as input the *hparam space* and returns a scalar.
   2. A *hyperopt method*: random search, grid, gradient-based, bayesian, genetic etc.
* Support inter-worker asynchronous and synchronous communication (?): MPI (?), nccl (?), sockets, collective operations (probably useful: AllGather, Broadcast) 

**Master-side** (maybe daemon?) responsibilities
---
 * Setting which hyperoptimization method and function to be loaded in the workers, also account for default procedures or automated selections according to the hyperparameters declared to be exported by worker's code.
 * Broadcasting other options and stuff to workers :P 
 * Distributed locking mechanism orchestration, if needed?
 * Resetting communicator worlds or managing what happens when a worker dies or effectively logs in.

Please complete any other thing missing from the aforementioned and comment on any of those. We have discussed whether it is possible to reuse some of the other hyperoptimization frameworks available, e.g. [hyperopt](https://github.com/hyperopt/hyperopt).

Other hyperoptimization software in general
---
 * [hyperopt](https://github.com/hyperopt/hyperopt)
 * [hype](https://github.com/hypelib/Hype)
 * [spearmint (old)](https://github.com/JasperSnoek/spearmint)
 * [spearmint](https://github.com/HIPS/Spearmint)
