*****
Oríon
*****

.. image:: https://travis-ci.org/mila-udem/orion.svg?branch=master
   :target: https://travis-ci.org/mila-udem/orion

.. image:: https://codecov.io/gh/mila-udem/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mila-udem/orion

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

Check out user's guide-101 for the simplest of demonstrations!

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
  Check `developer's guide-101`_. Expect algorithm plugins to pop out quickly!
- Came up with an idea?
  Your intuition is still at play:
  Help your optima hunter now by a command line interface.
- And other many more already there or coming soon!

.. _MongoDB: https://www.mongodb.com/
.. _Skopt: https://scikit-optimize.github.io/
.. _developer's guide-101: /developer/testing

Installation
============

Install Oríon (beta) by running:

``pip install git+https://github.com/mila-udem/orion.git@master``

For more information read the full installation docs.
.. TODO Once website is published use a full hyperlink to the installation page

Contribute or Ask
=================

Do you have a question or issues?
Do you want to report a bug or suggest a feature? Name it!
Please contact us by opening an issue in our repository below:

- Issue Tracker: `github.com/mila-udem/orion/issues`
- Source Code: `github.com/mila-udem/orion`

Start by starring and forking our Github repo!

Thanks for the support!

License
=======

The project is licensed under the BSD license.
