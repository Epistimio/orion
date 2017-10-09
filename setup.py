#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versioneer

setup(name='metaopt',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Hyperparameter Optimization',
      long_description=open('README.md').read(),
      license='GNU General Public License v3.0',
      author='MetaOptimizer Team - MILA/Université de Montréal',
      author_email='lisa_labo@iro.umontreal.ca',
      url='https://github.com/mila-udem/metaopt',
      packages=[],
      scripts=[],
      install_requires=[],
      )
