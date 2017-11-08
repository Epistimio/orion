#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import os
import textwrap
from glob import iglob

from setuptools import (setup, find_packages)

import versioneer

isfile = os.path.isfile
pjoin = os.path.join
repo_root = os.path.dirname(os.path.abspath(__file__))
mpath = pjoin(repo_root, 'src')
sys.path.insert(0, mpath)

import metaopt


def find_data_files():
    """Find MetaOpt's configuration and metadata files"""
    install_config_path = pjoin(metaopt.dirs.site_data_dir, 'config')
    config_path = pjoin('config', '*')
    configs = [cfg for cfg in iglob(config_path) if isfile(cfg)]

    data_files = [
        (install_config_path, configs),
        (metaopt.dirs.site_data_dir, ['LICENSE', 'README.md']),
        ]

    return data_files


setup_args = dict(
    name=metaopt.__name__,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=metaopt.__descr__,
    long_description=textwrap.dedent(metaopt.__doc__),
    license=metaopt.__license__,
    author=metaopt.__author__,
    author_email=metaopt.__author_email__,
    url=metaopt.__url__,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    data_files=find_data_files(),
    scripts=['scripts/mopt'],
    install_requires=['six', 'PyYAML'],
    setup_requires=[],
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
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]


if __name__ == '__main__':
    setup(**setup_args)
