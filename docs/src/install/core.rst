****************************
Installation of Orion's core
****************************

Oríon should work on most Linux distributions and Mac OS X. It is tested on Ubuntu 16.04 LTS and Mac
OS X 10.13. We do not support Windows and there is no short term plan to do so.

Via PyPI
========

The easiest way to install Oríon is using the Python package manager. The core of Oríon is
registered on PyPI under `orion`.

.. code-block:: sh

   pip install orion

This will install all the core components. Note that the only algorithm provided with it
is random search. To install more algorithms, you can look at section :doc:`/install/plugins`.

Via Git
=======

This way is recommended if you want to work with the bleeding edge version
of Oríon.

.. code-block:: sh

   pip install git+https://github.com/epistimio/orion.git@develop

Note that the bleeding-edge branch is develop. The master branch is the same as the latest version
on PyPI.
