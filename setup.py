#!/usr/bin/env python
from setuptools import setup, Extension, distutils, Command, find_packages
from distutils.core import setup
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.unixccompiler
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os

DEBUG = check_env_flag('DEBUG')


class clean(distutils.command.clean.clean):

    def run(self):
        import glob
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(bool, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

################################################################################
# Configuring compile flags
################################################################################

include_dirs = []
library_dirs = []
extra_link_args = []
extra_compile_args = ['-std=c++1z', '-Wno-write-strings',
                      # Python 2.6 requires -fno-strict-aliasing, see
                      # http://legacy.python.org/dev/peps/pep-3123/
                      '-fno-strict-aliasing']

cmdclass = {
    'build': build,
    'develop': develop,
    'install': install,
    'clean': clean,
}

cmdclass.update(build_dep_cmds)

setup(name='Metaopt',
      version='0.1',
      description='Meta Hyperparameter Optimization',
      author='MetaOptimizer Team - Universite de Montreal',
      author_email='lisa_labo@iro.umontreal.ca',
      url='https://mila.quebec',
      packages=['distutils', 'distutils.command'],
      cmdclass=cmdclass,
      install_requires=['pyyaml', 'numpy'],
     )
