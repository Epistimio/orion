#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for MetaOpt."""
from __future__ import absolute_import

from glob import iglob
import os
import sys
import textwrap

from setuptools import (find_packages, setup)

import versioneer

isfile = os.path.isfile
pjoin = os.path.join
repo_root = os.path.dirname(os.path.abspath(__file__))
mpath = pjoin(repo_root, 'src')
sys.path.insert(0, mpath)

import metaopt.core as metaopt  # noqa


def find_data_files():
    """Find MetaOpt's configuration and metadata files."""
    install_config_path = pjoin(metaopt.DIRS.site_data_dir, 'config')
    config_path = pjoin('config', '*')
    configs = [cfg for cfg in iglob(config_path) if isfile(cfg)]

    data_files = [
        (install_config_path, configs),
        (metaopt.DIRS.site_data_dir, ['LICENSE', 'README.rst']),
    ]

    return data_files


setup_args = dict(
    name='metaopt',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=metaopt.__descr__,
    long_description=textwrap.dedent(metaopt.__doc__),
    license=metaopt.__license__,
    author=metaopt.__author__,
    author_email=metaopt.__author_email__,
    url=metaopt.__url__,
    namespace_packages=['metaopt'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    data_files=find_data_files(),
    entry_points={
        'console_scripts': [
            'mopt = metaopt.core.cli:main',
            ],
        },
    install_requires=['six', 'PyYAML', 'pymongo>=3'],
    tests_require=['pytest>=3.0.0'],
    setup_requires=['setuptools', 'pytest-runner>=2.0,<3dev'],
    #  http://peak.telecommunity.com/DevCenter/setuptools#setting-the-zip-safe-flag
    #  zip_safe=False
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
