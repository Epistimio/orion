*********************************************
PyTorch A2C PPO ACKTR
*********************************************

.. note ::

    If Oríon not installed: pip install orion


Intro
=====

Here, we are looking to update the ikostrikov/pytorch-a2c-ppo-acktr
Reinforcement Learning algorithm implementations to use Oríon to find the best
hyperparameters while trying to prevent overfitting via a validation set of
random evaluation seeds in the environment.

What to change
==============


To get the original repository of `ikostrikov`_
to work using Orion, we make a couple of changes.

.. _ikostrikov: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

First, we fork the original repo at commit hash:
4d95ec364c7303566c6a52fb0a254640e931609d

To the top of

.. code-block:: bash

    main.py

we add:

.. code-block:: python

    from orion.client import report_objective

Then, we ensure that we evaluate on a separate set of hold out random seeds for
the environment (which should be different than the test set and training seed).
For MuJoCo environments where the random seed has an effect, we can simply set
the random seed before a rollout. In Atari, we would have to create a new
validation set of rollouts perhaps with different human starts.

The original repository doesn't separate training/validation/testing so we add
the required methods as follows. We create a file with functions for evaluation:

.. code-block:: bash

    eval.py

.. note ::

  The execution with Oríon does not require the added evaluation methods for
  a validation set and could use the final training performance. However, for
  sake of adhering to best practices, we create a validation set method in
  eval.py.

And then simply add a registration for the evaluation after training our
algorithm:

.. code-block:: python

    validation_returns = evaluate_with_seeds(eval_env,
                                             actor_critic,
                                             args.cuda,
                                             eval_env_seeds)

    report_objective(name='validation_return', objective=np.mean(validation_returns))

Now we're ready to go to run orion's hyperparameter optimization!

How to search for hyperparameters
=================================

.. code-block:: bash

  orion -v hunt -n ppo_hopper \
    python main.py --env-name "Hopper-v2" --algo ppo --use-gae --vis-interval 1 \
    --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 \
    --lr~'loguniform(1e-5, 1.0)' --entropy-coef 0 --value-loss-coef 1 \
    --ppo-epoch 10 --num-mini-batch 32 --gamma~'uniform(.95, .9995)' --tau 0.95 \
    --num-frames 1000000 --eval-env-seeds-file ./seeds.json --no-vis \
    --log-dir~trial.hash_name

Notice that this will search over the learning rates and gamma values,
while setting the log directory name to be the hashed trial name provided
in the orion database.

The full modified codebase for use with Oríon can be found on Gihub:

.. code-block:: bash

    git clone https://github.com/Breakend/orion-pytorch-ppo-acktr-a2c
