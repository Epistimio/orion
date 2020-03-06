.. contents:: Developer's Guide 103: Contribution Standards

**********
Contribute
**********

Coding and Repository Standards
===============================

We are using flake8_ (along with some of its plugins) and pylint_.
Their styles are provided in ``/tox.ini`` and ``/.pylintrc`` respectively.

.. code-block:: sh

   tox -e flake8
   tox -e pylint

Also, we are using a check-manifest_ which compares ``/MANIFEST.in`` and git
structure of the source repository, and finally readme_renderer_ which
checks whether ``/README.rst`` can be
actually rendered in PyPI_ website page for Oríon.

.. code-block:: sh

   tox -e packaging

To run all of expected linters execute::

   tox -e lint

.. _flake8: http://flake8.pycqa.org/en/latest/
.. _pylint: https://www.pylint.org/
.. _check-manifest: https://pypi.org/project/check-manifest/
.. _readme_renderer: https://pypi.org/project/readme_renderer/
.. _PyPI: https://pypi.org/

Fork and Pull Request
=====================

Fork Oríon remotely to your Github_ account now, and start by submitting a
`Pull Request <https://github.com/epistimio/orion/pulls>`_ to us or by
discussing an `issue <https://github.com/epistimio/orion/issues>`_ with us.

.. image:: https://img.shields.io/github/forks/epistimio/orion.svg?style=social&label=Fork
   :target: https://github.com/epistimio/orion/network

.. _Github: https://github.com
