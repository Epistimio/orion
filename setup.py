#!/usr/bin/env python
"""Installation script for Oríon."""
import os

from setuptools import setup

import versioneer

repo_root = os.path.dirname(os.path.abspath(__file__))

with open("tests/requirements.txt") as f:
    tests_require = f.readlines()

packages = [  # Packages must be sorted alphabetically to ease maintenance and merges.
    "orion.algo",
    "orion.algo.mofa",
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
    "docs": [
        "matplotlib",
        "numpydoc",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinxcontrib.httpdomain",
        "sphinx-autoapi",
        "sphinx_gallery",
    ],
    "dask": ["dask[complete]"],
    "track": ["track @ git+https://github.com/Delaunay/track@master#egg=track"],
    "profet": ["emukit", "GPy", "torch", "pybnn"],
    "configspace": ["ConfigSpace"],
    "ax": [
        "ax-platform",
        "numpy",
    ],
    "dehb": [
        "ConfigSpace",
        "dehb @ git+https://github.com/bouthilx/DEHB.git@master#egg=dehb",
        "sspace @ git+https://github.com/Epistimio/sample-space.git@master#egg=sspace",
    ],
    "bohb": [
        "hpbandster",
        "ConfigSpace",
        "sspace @ git+https://github.com/Epistimio/sample-space.git@master#egg=sspace",
    ],
    "pb2": ["GPy"],
    "nevergrad": ["nevergrad>=0.4.3.post10", "fcmaes", "pymoo"],
    "hebo": [
        "numpy",
        "pymoo==0.5.0",
        "hebo @ git+https://github.com/huawei-noah/HEBO.git@v0.3.2#egg=hebo&subdirectory=HEBO",
    ],
}
extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

dashboard_files = []
for root, sub_directories, files in os.walk("dashboard/build"):
    for file in files:
        # Install dashboard build in a folder `orion-dashboard`
        install_root = os.path.join("orion-dashboard", *root.split(os.sep)[1:])
        dashboard_files.append((install_root, [os.path.join(root, file)]))

setup_args = dict(
    name="orion",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Asynchronous [black-box] Optimization",
    long_description=open(
        os.path.join(repo_root, "README.rst"), encoding="utf8"
    ).read(),
    license="BSD-3-Clause",
    author="Epistímio",
    author_email="xavier.bouthillier@umontreal.ca",
    url="https://github.com/epistimio/orion",
    packages=packages,
    package_dir={"": "src"},
    data_files=dashboard_files,
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
            # "pbt = orion.algo.pbt.pbt:PBT",
            "ax = orion.algo.axoptimizer:AxOptimizer",
            "mofa = orion.algo.mofa.mofa:MOFA",
            "pb2 = orion.algo.pbt.pb2:PB2",
            "bohb = orion.algo.bohb:BOHB",
            "nevergrad = orion.algo.nevergradoptimizer:NevergradOptimizer",
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
        "scikit-optimize",
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
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)
