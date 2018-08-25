.. contents:: User's Guide 102: Problem Definition

*************
Define Search
*************

I intend to support extensible plugin support for problem's definition for both
CLI inputs and configuration file types.

Parameters by CLI
=================

We support currently only argparse_ arguments, like::

   orion -v -n demo tests/functional/demo/black_box.py -x~'normal(30, 3)'

The list of supported distributions can be found [here](https://docs.scipy.org/doc/scipy/reference/stats.html).

Parameters by configuration files
=================================

In addition, one can specify parameter search dimension within configuration
files of popular type, which will be used to pass input values to the script
whose execution Oríon targets to optimize.

Currently we support YAML_ or JSON_ file types.

Here are example files which correspond to the same configuration:

   * `YAML example <https://github.com/epistimio/orion/blob/master/tests/unittests/core/sample_config.yml>`_
   * `JSON example <https://github.com/epistimio/orion/blob/master/tests/unittests/core/sample_config.json>`_

The configuration file passed as an input template and as problem domain
definition must be either **the first positional argument** accepted by an
executable to be optimize or **specified by ``--config`` keyword argument**.

.. _argparse: https://docs.python.org/3.6/library/argparse.html
.. _YAML: http://yaml.org/
.. _JSON: https://www.json.org/
