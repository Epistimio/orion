*****
Oríon
*****

|pypi| |py_versions| |license| |doi|
|rtfd| |codecov| |github-actions|

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

.. |rtfd| image:: https://readthedocs.org/projects/orion/badge/?version=stable
    :target: https://orion.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/Epistimio/orion/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Epistimio/orion
    :alt: Codecov Report

.. |github-actions| image:: https://github.com/Epistimio/orion/workflows/build/badge.svg?branch=master&event=pull_request
    :target: https://github.com/Epistimio/orion/actions?query=workflow:build+branch:master+event:schedule
    :alt: Github actions tests

.. image:: _static/logos/orion_logo_grid_150ppi.png
  :width: 400
  :alt: Oríon

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

Check out our `getting started guide`_ or `this presentation
<https://bluejeans.com/playback/s/4WUezzFCmb9StHzYgB0RjVbTUCKnRcptBvzBMP7t2UpLyKuAq7Emieo911BqEMnI>`_
for an overview, or our `scikit-learn example`_ for a more hands-on experience. Finally we
encourage you to browse our `documentation`_.

.. _getting started guide: https://orion.readthedocs.io/en/stable/install/gettingstarted.html
.. _documentation: https://orion.readthedocs.io/
.. _scikit-learn example: https://orion.readthedocs.io/en/stable/tutorials/scikit-learn.html

Why Oríon?
==========

*Effortless to adopt, deeply customizable*

- `Adopt it <https://orion.readthedocs.io/en/stable/user/script.html>`_ with a single line of code
- `Natively asynchronous <https://orion.readthedocs.io/en/stable/code/core.html>`_, thus resilient and easy to parallelize
- Offers the latest established hyperparameter `algorithms <https://orion.readthedocs.io/en/stable/user/algorithms.html>`_
- Elegant and rich `search-space definitions <https://orion.readthedocs.io/en/stable/user/searchspace.html>`_
- Comprehensive `configuration <https://orion.readthedocs.io/en/stable/user/config.html>`_ system with smart defaults
- Transparent persistence in local or remote `database <https://orion.readthedocs.io/en/stable/install/database.html>`_
- `Integrate seamlessly <https://orion.readthedocs.io/en/stable/plugins/base.html>`_ your own
  hyper-optimization algorithms
- `Language <https://orion.readthedocs.io/en/stable/user/script.html#language-compatibility>`_
  and `configuration file <https://orion.readthedocs.io/en/stable/user/searchspace.html#configuration-file>`_ agnostic

Installation
============

Install Oríon by running ``$ pip install orion``. For more information consult the `installation
guide`_.

.. _installation guide: https://orion.readthedocs.io/en/stable/install/core.html

Presentations
=============

- 2021-07-14 - SciPy 2021 (`Video <https://youtu.be/H1jQBQIbQmA>`__) (`Slides <https://docs.google.com/presentation/d/1Aekt1hKtiT1y7pyvWeRRoVND4cDUYFj53xAzk8Zb8z0/edit?usp=sharing>`__)
- 2021-05-19 - Dask Summit 2021 (`Video <https://youtu.be/W5oWdRiSSr8>`__) (`Slides <https://docs.google.com/presentation/d/1MBy0gjWNV-8GjtEXVLCThN8JddK7znHSg7puycwkBZ4/edit?usp=sharing>`__)
- 2021-03-16 - AICamp
  (`Video
  <https://www.youtube.com/watch?v=QQ69vxF3LTI>`__)
  (`Slides
  <https://docs.google.com/presentation/d/1Tq3KrWcp66wdlZJtCFaxfq1m5ydyhcPiDCGCOuh_REg/edit?usp=sharing>`__)
- 2019-11-28 - Tech-talk @ Mila
  (`Video
  <https:/bluejeans.com/playback/s/4WUezzFCmb9StHzYgB0RjVbTUCKnRcptBvzBMP7t2UpLyKuAq7Emieo911BqEMnI>`__)
  (`Slides
  <https://docs.google.com/presentation/d/18g7Q4xRuhMtcVbwmFwDfH7v9gKS252-laOi9HrEQ7a4/edit?usp=sharing>`__)

Contribute or Ask
=================

Do you have a question or issues?
Do you want to report a bug or suggest a feature? Name it!
Please contact us by opening an issue in our repository below and checkout our `contribution guidelines <https://github.com/Epistimio/orion/blob/develop/CONTRIBUTING.md>`_:

- Issue Tracker: `<https://github.com/epistimio/orion/issues>`_
- Source Code: `<https://github.com/epistimio/orion>`_

Start by starring and forking our Github repo!

Thanks for the support!

Citation
========

If you use Oríon for published work, please cite our work using the following bibtex entry.

.. code-block:: bibtex

   @software{xavier_bouthillier_2023_0_2_7,
     author       = {Xavier Bouthillier and
                     Christos Tsirigotis and
                     François Corneau-Tremblay and
                     Thomas Schweizer and
                     Lin Dong and
                     Pierre Delaunay and
                     Fabrice Normandin and
                     Mirko Bronzi and
                     Dendi Suhubdy and
                     Reyhane Askari and
                     Michael Noukhovitch and
                     Chao Xue and
                     Satya Ortiz-Gagné and
                     Olivier Breuleux and
                     Arnaud Bergeron and
                     Olexa Bilaniuk and
                     Steven Bocco and
                     Hadrien Bertrand and
                     Guillaume Alain and
                     Dmitriy Serdyuk and
                     Peter Henderson and
                     Pascal Lamblin and
                     Christopher Beckham},
     title        = {{Epistimio/orion: Asynchronous Distributed Hyperparameter Optimization}},
     month        = march,
     year         = 2023,
     publisher    = {Zenodo},
     version      = {v0.2.7,
     doi          = {10.5281/zenodo.3478592},
     url          = {https://doi.org/10.5281/zenodo.3478592}
   }

Roadmap
=======

See `ROADMAP.md <https://github.com/Epistimio/orion/blob/master/ROADMAP.md>`_.

License
=======

The project is licensed under the `BSD license <https://github.com/Epistimio/orion/blob/master/LICENSE>`_.
