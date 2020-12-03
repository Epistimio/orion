#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for `orion.algo.gradient_descent`."""
from setuptools import setup

setup_args = dict(
    name="orion.algo.gradient_descent",
    version=0.1,
    description="Implement a gradient descent algorithm, for demo and testing.",
    license="BSD-3-Clause",
    author="Christos Tsirigotis",
    author_email="tsirif@gmail.com",
    url="https://github.com/epistimio/orion",
    packages=["orion.algo"],
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "OptimizationAlgorithm": [
            "gradient_descent = orion.algo.gradient_descent:Gradient_Descent"
        ],
    },
    install_requires=["orion"],
    setup_requires=["setuptools"],
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
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.6 3.7 3.8".split()]

if __name__ == "__main__":
    setup(**setup_args)
