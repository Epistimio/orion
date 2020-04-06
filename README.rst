*****
Oríon
*****

|pypi| |py_versions| |license| |doi|
|rtfd| |codecov| |travis|

.. |pypi| image:: https://img.shields.io/pypi/v/orion.svg
    :target: https://pypi.python.org/pypi/orion
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/orion.svg
    :target: https://pypi.python.org/pypi/orion
    :alt: Supported Python Versions

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: BSD 3-clause license

.. |doi| image:: https://zenodo.org/badge/102697867.svg
   :target: https://zenodo.org/badge/latestdoi/102697867
   :alt: DOI

.. |rtfd| image:: https://readthedocs.org/projects/orion/badge/?version=latest
    :target: https://orion.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/Epistimio/orion/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Epistimio/orion
    :alt: Codecov Report

.. |travis| image:: https://travis-ci.org/Epistimio/orion.svg?branch=master
    :target: https://travis-ci.org/Epistimio/orion
    :alt: Travis tests

Oríon is an asynchronous framework for black-box function optimization.

Its purpose is to serve as a meta-optimizer for machine learning models
and training, as well as a flexible experimentation
platform for large scale asynchronous optimization procedures.

Core design value is the minimum disruption of a researcher's workflow.
It allows fast and efficient tuning, providing minimum simple non-intrusive
(not even necessary!) helper *client* interface for a user's script.

So if ``./run.py --mini-batch=50`` looks like what you execute normally,
now what you have to do looks like this:

``orion -n experiment_name ./run.py --mini-batch~'randint(32, 256)'``

Check out our `pytorch example`_ for a quick demonstration!

.. _pytorch example: https://orion.readthedocs.io/en/latest/tutorials/pytorch.html

Features
========
*As simple and as complex you want*

- Simple and natural, but also explicit and verbose, search domain definitions
- Minimal and non-intrusive client interface for reporting
  target function values
- Database logging (currently powered by MongoDB_)
- Flexible configuration
- Explicit experiment termination conditions
- Algorithms algorithms algorithms:
  Skopt_'s bayesian optimizers are at hand without writing.
  Random search is the default.
  **only** a single line of code.
- More algorithms:
  Implementing and distributing algorithms is as easy as possible!
  Check the `plugin documentation`_. Expect algorithm plugins to pop out quickly!
- Came up with an idea?
  Your intuition is still at play:
  Help your optima hunter now by a command line interface.
- And other many more already there or coming soon!

.. _MongoDB: https://www.mongodb.com/
.. _Skopt: https://scikit-optimize.github.io/
.. _plugin documentation: https://orion.readthedocs.io/en/latest/plugins/base.html

Installation
============

Install Oríon by running:

``pip install orion``

For more information read the `full installation docs`_.

.. _full installation docs: https://orion.readthedocs.io/en/latest/install/core.html

Contribute or Ask
=================

Do you have a question or issues?
Do you want to report a bug or suggest a feature? Name it!
Please contact us by opening an issue in our repository below and checkout our `contribution guidelines <https://github.com/Epistimio/orion/blob/master/CONTRIBUTING.md>`_:

- Issue Tracker: `<https://github.com/epistimio/orion/issues>`_
- Source Code: `<https://github.com/epistimio/orion>`_

Start by starring and forking our Github repo!

Thanks for the support!

Roadmap
========

You can find our roadmap here: `<https://github.com/Epistimio/orion/blob/master/ROADMAP.md>`_

Citation
========

If you use Oríon for published work, please cite our work using the following bibtex entry.

.. code-block:: bibtex

   @software{xavier_bouthillier_2019_3478593,
     author       = {Xavier Bouthillier and
                     Christos Tsirigotis and
                     François Corneau-Tremblay and
                     Pierre Delaunay and
                     Reyhane Askari and
                     Dendi Suhubdy and
                     Michael Noukhovitch and
                     Dmitriy Serdyuk and
                     Arnaud Bergeron and
                     Peter Henderson and
                     Pascal Lamblin and
                     Mirko Bronzi and
                     Christopher Beckham},
     title        = {Oríon - Asynchronous Distributed Hyperparameter Optimization},
     month        = oct,
     year         = 2019,
     publisher    = {Zenodo},
     version      = {v0.1.7},
     doi          = {10.5281/zenodo.3478592},
     url          = {https://doi.org/10.5281/zenodo.3478592}
   }

License
=======

The project is licensed under the BSD license.
