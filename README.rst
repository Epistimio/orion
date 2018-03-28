Orion
=======

.. image:: https://travis-ci.org/mila-udem/orion.svg?branch=master
   :target: https://travis-ci.org/mila-udem/orion

.. image:: https://codecov.io/gh/mila-udem/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mila-udem/orion

Orion is an asynchronous framework for black-box function optimization.

Its purpose is to serve as a meta-optimizer for machine learning models
and training, as well as a flexible experimentation
platform for large scale asynchronous optimization procedures.

Core design value is the minimum disruption of a researcher's workflow.
It allows fast and efficient tuning, providing minimum simple non-intrusive
(not even necessary!) helper *client* interface for a user's script.

So if ``./run.py --mini-batch=50`` looks like what you execute normally,
now what you have to do looks like this:

``mopt -n experiment_name ./run.py --mini-batch~'randint(32, 256)'``

Check out user's guide-101 for the simplest of demonstrations!

Features
--------
*As simple and as complex you want*

- Simple and natural, but also explicit and verbose, search domain definitions
- Minimal and non-intrusive client interface for reporting
  target function values
- Database logging (currently powered by MongoDB)
- Flexible configuration
- Explicit experiment termination conditions
- Algorithms algorithms algorithms:
  Skopt's bayesian optimizers are at hand without writing.
  Random search is the default.
  **only** a single line of code.
- More algorithms:
  Implementing and distributing algorithms is as easy as possible!
  Check developer's guide-201. Expect algorithm plugins to pop out quickly!
- Came up with an idea?
  Your intuition is still at play:
  Help your optima hunter now by a command line interface.
- And other many more already there or coming soon!

Installation
------------

Install Orion (beta) by running:

``pip install git+https://github.com/mila-udem/orion.git@master``

Contribute
----------

Do you have a question?

Do you want to report a bug or suggest a feature? Name it!

- Issue Tracker: github.com/mila-udem/orion/issues
- Source Code: github.com/mila-udem/orion

Support
-------

If you are having issues, contact us immediately.
We have a mailing list located at: voici@voilà.com

License
-------

The project is licensed under the BSD license.


Core Developers
---------------

Christos Tsirigotis [`@tsirif <https://github.com/tsirif>`_]

Xavier Bouthillier [`@bouthilx <https://github.com/bouthilx>`_]
