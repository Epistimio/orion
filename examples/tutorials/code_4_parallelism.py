"""
.. _executor_backends:

***************
Scaling workers
***************

This tutorial shows simple examples to use the different backends of Oríon to scale execution to
multiple workers in parallel.
The parallelization of Oríon workers is explained in more details in section :ref:`parallelism`.

We will start with a basic example using scikit-learn.
"""
import joblib
import numpy
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

# flake8: noqa: E265


def main(C, gamma, tol, class_weight, joblib_backend="loky"):

    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    model = SVC(kernel="rbf", C=C, gamma=gamma, tol=tol, class_weight=class_weight)

    # Single metric evaluation using cross_validate
    with joblib.parallel_backend(joblib_backend):
        cv_results = cross_validate(model, X, y, cv=5)

    accuracy = numpy.mean(cv_results["test_score"])
    error_rate = 1 - accuracy

    return [{"name": "test_error_rate", "type": "objective", "value": error_rate}]


#%%
# We create a ``main()`` function that takes as arguments the hyperparameters for an SVM for
# classification. We add the argument ``joblib_backend`` so that we can control which backend
# is used to paralellize the cross-validation. By default we will use ``loky`` for multi-processing.
# We load the dataset ``digits`` and divide it in features ``X`` and targets ``y``.
# We then create the model with the given hyperparameter values.
# We use ``joblib.parallel_backend`` to execute the cross-validation on 5 folds in parallel
# for more efficiency. Finally we compute the average accuracy, and convert it to test error
# rate since Oríon is minimizing the objective and we would not want to minimize the accuracy.
# The results is returned in the format required by Oríon.
#
# We will now create an experiment with Oríon to optimize this function.

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "./db.pkl",
    },
}

# Specify optimization space for the SVM
space = {
    "C": "loguniform(1e-6, 1e6)",
    "gamma": "loguniform(1e-8, 1e8)",
    "tol": "loguniform(1e-4, 1e-1)",
    "class_weight": "choices([None, 'balanced'])",
}


#%%
# Joblib
# ------
#
# `Joblib`_ is a lightweight library for task parallel execution in Python. It is the default
# backend used by Oríon to spawn multiple workers.
#
# We first build the experiment and limit it to 200 trials.

from orion.client import build_experiment

experiment = build_experiment(
    name="joblib_example",
    max_trials=200,
    space=space,
    storage=storage,
)

#%%
# Since joblib is the default backend, we do not need to do anything special to use it.
# We can simply call
# :meth:`ExperimentClient.workon() <orion.client.experiment.ExperimentClient.workon>`
# and specify the number of workers that we want.


experiment.workon(main, n_workers=4)

#%%
# It is as simple as this.
#
# The experiment backend is by default the one configured in the global configuration
# (:ref:`config_worker_executor` and :ref:`config_worker_executor_configuration`).
# If you want to use a different backend while executing
# :meth:`ExperimentClient.workon() <orion.client.experiment.ExperimentClient.workon>`, you can use
# :meth:`ExperimentClient.tmp_executor() <orion.client.experiment.ExperimentClient.tmp_executor>`
# like the following.


with experiment.tmp_executor("joblib", n_workers=10):
    experiment.workon(main, n_workers=2)

#%%
# Note that you must specify ``n_workers`` for both the backend and for
# :meth:`ExperimentClient.workon() <orion.client.experiment.ExperimentClient.workon>`.
# For the backend, it
# refers to the number of workers you would like to have running in parallel.
# For :meth:`ExperimentClient.workon() <orion.client.experiment.ExperimentClient.workon>`,
# it is the number of
# Orion workers to run in parallel. You may want to have more backend workers than
# Oríon workers if for instance your task is also parallelizing tasks. You can
# see such an example here, because we are parallelizing the cross-validation inside
# the function Oríon is optimizing. Each worker will create 5 tasks that can be run in parallel.
#
# .. _Joblib: https://joblib.readthedocs.io/en/latest/
#
# .. _parallel_backend(): https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend
#
# Dask
# ----
#
# Using Dask is similar to joblib. Unless you have configured
# Oríon to use Dask by default
# (:ref:`config_worker_executor` and :ref:`config_worker_executor_configuration`),
# you will want to use
# :meth:`ExperimentClient.tmp_executor() <orion.client.experiment.ExperimentClient.tmp_executor>`
# to change the backend. Make sure to only run Dask with if ``__name__ == "__main__"``
# otherwise you will run into a RuntimeError
# (`see here <https://github.com/dask/distributed/issues/2520>`_).

experiment = build_experiment(
    name="dask_example",
    max_trials=200,
    space=space,
    storage=storage,
)

if __name__ == "__main__":
    with experiment.tmp_executor("dask", n_workers=10):
        experiment.workon(main, n_workers=2, joblib_backend="dask")
