#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for `metaopt.algo.gradient_descent`."""
from __future__ import absolute_import

from setuptools import (find_packages, setup)

setup_args = dict(
    name='metaopt.algo.gradient_descent',
    version=0.1,
    description="Implement a gradient descent algorithm, for demo and testing.",
    license='BSD-3-Clause',
    author='Christos Tsirigotis',
    author_email='tsirif@gmail.com',
    url='https://github.com/mila-udem/metaopt',
    namespace_packages=['metaopt', 'metaopt.algo'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'OptimizationAlgorithm': [
            'gradient_descent = metaopt.algo.gradient_descent:Gradient_Descent'
            ],
        },
    install_requires=['metaopt.core'],
    setup_requires=['setuptools', 'metaopt.core'],
    )

setup_args['keywords'] = [
    'Machine Learning',
    'Deep Learning',
    'Distributed',
    'Optimization',
    ]

setup_args['platforms'] = ['Linux']

setup_args['classifiers'] = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
] + [('Programming Language :: Python :: %s' % x)
     for x in '2 2.6 2.7 3 3.4 3.5 3.6'.split()]

if __name__ == '__main__':
    setup(**setup_args)
