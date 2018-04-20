# -*- coding: utf-8 -*-
"""
MetaOpt is an asynchronous distributed framework for black-box function optimization.

Its purpose is to serve as a hyperparameter optimizer for
machine learning models and training, as well as a flexible experimentation
platform for large scale asynchronous optimization procedures.

It has been designed firstly to disrupt a user's workflow at minimum, allowing
fast and efficient hyperparameter tuning, and secondly to provide secondary APIs
for more advanced features, such as dynamically reporting validation scores on
training time for automatic early stopping or on-the-fly reconfiguration.

Start by having a look here: https://github.com/mila-udem/orion
"""
from ._version import get_versions
from .utils._appdirs import AppDirs

VERSIONS = get_versions()
del get_versions

__descr__ = 'Distributed Asynchronous [black-box] Optimization'
__version__ = VERSIONS['version']
__license__ = 'BSD-3-Clause'
__author__ = 'MetaOptimizer Team - MILA, Université de Montréal'
__author_short__ = 'MILA'
__author_email__ = 'lisa_labo@iro.umontreal.ca'
__authors__ = {
    'tsirif': ('Christos Tsirigotis', 'tsirif@gmail.com'),
}
__url__ = 'https://github.com/mila-udem/orion'

DIRS = AppDirs(__name__, __author_short__)
del AppDirs
