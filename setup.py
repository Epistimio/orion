#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for Oríon."""
import os

from setuptools import setup

import versioneer


repo_root = os.path.dirname(os.path.abspath(__file__))

with open("tests/requirements.txt") as f:
    tests_require = f.readlines()

packages = [  # Packages must be sorted alphabetically to ease maintenance and merges.
    "orion.algo",
    "orion.algo.dehb",
    "orion.algo.pbt",
    "orion.analysis",
    "orion.benchmark",
    "orion.client",
    "orion.core",
    "orion.executor",
    "orion.plotting",
    "orion.serving",
    "orion.storage",
    "orion.testing",
]

extras_require = {
    "test": tests_require,
    "dask": ["dask[complete]"],
    "track": ["track @ git+https://github.com/Delaunay/track"],
    "profet": ["emukit", "GPy", "torch", "pybnn"],
    "dehb": [
        "ConfigSpace",
        "dehb @ git+https://github.com/automl/DEHB.git@development",
        "sspace @ git+https://github.com/Epistimio/sample-space.git",
    ],
}
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup_args = dict(
    name="orion",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Asynchronous [black-box] Optimization",
    long_description=open(
        os.path.join(repo_root, "README.rst"), "rt", encoding="utf8"
    ).read(),
    license="BSD-3-Clause",
    author="Epistímio",
    author_email="xavier.bouthillier@umontreal.ca",
    url="https://github.com/epistimio/orion",
    packages=packages,
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "orion = orion.core.cli:main",
        ],
        "BaseAlgorithm": [
            "random = orion.algo.random:Random",
            "gridsearch = orion.algo.gridsearch:GridSearch",
            "hyperband = orion.algo.hyperband:Hyperband",
            "asha = orion.algo.asha:ASHA",
            "dehb = orion.algo.dehb.dehb:DEHB",
            "tpe = orion.algo.tpe:TPE",
            "EvolutionES = orion.algo.evolution_es:EvolutionES",
            "pbt = orion.algo.pbt.pbt:PBT",
        ],
        "Database": [
            "ephemeraldb = orion.core.io.database.ephemeraldb:EphemeralDB",
            "pickleddb = orion.core.io.database.pickleddb:PickledDB",
            "mongodb = orion.core.io.database.mongodb:MongoDB",
        ],
        "BaseStorageProtocol": [
            "track = orion.storage.track:Track",
            "legacy = orion.storage.legacy:Legacy",
        ],
        "BaseExecutor": [
            "singleexecutor = orion.executor.single_backend:SingleExecutor",
            "joblib = orion.executor.joblib_backend:Joblib",
            "dask = orion.executor.dask_backend:Dask",
        ],
    },
    install_requires=[
        "cloudpickle",
        "PyYAML",
        "pymongo>=3",
        "numpy",
        "scipy",
        "gitpython",
        "filelock",
        "tabulate",
        "AppDirs",
        "plotly",
        "kaleido",
        "requests",
        "pandas",
        "gunicorn",
        "falcon",
        "falcon-cors",
        "scikit-learn",
        "psutil",
        "joblib",
        "pytest>=3.0.0",
    ],
    tests_require=tests_require,
    setup_requires=["setuptools", "appdirs", "pytest-runner"],
    extras_require=extras_require,
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False,
)

setup_args["keywords"] = [
    "Machine Learning",
    "Deep Learning",
    "Distributed",
    "Optimization",
]

setup_args["platforms"] = ["Linux"]

setup_args["classifiers"] = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.7 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)
