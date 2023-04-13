"""
===========
Parallelism
===========


Multiprocessing
---------------

(joblib's loky)


Dask
----


"""

import joblib
import numpy
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

from orion.client import create_experiment


def main(C, gamma, tol, class_weight):

    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target

    model = SVC(kernel="rbf", C=C, gamma=gamma, tol=tol, class_weight=class_weight)

    # Single metric evaluation using cross_validate
    with joblib.parallel_backend("dask"):
        cv_results = cross_validate(model, X, y, cv=5)

    accuracy = numpy.mean(cv_results["test_score"])
    error_rate = 1 - accuracy

    return [{"name": "test_error_rate", "type": "objective", "value": error_rate}]


def hpo(n_workers=16):

    experiment = create_experiment(
        name="dask",
        max_trials=1000,
        max_broken=5,
        space={
            "C": "loguniform(1e-6, 1e6, precision=None)",
            "gamma": "loguniform(1e-8, 1e8, precision=None)",
            "tol": "loguniform(1e-4, 1e-1, precision=None)",
            "class_weight": "choices([None, 'balanced'])",
        },
        algorithm={"random": {"seed": 1}},
    )

    with experiment.tmp_executor("dask", n_workers=n_workers):
        experiment.workon(main, n_workers=n_workers // 2)

    experiment.plot.regret().show()
    experiment.plot.partial_dependencies(params=["C", "gamma", "tol"]).show()


if __name__ == "__main__":
    hpo()
