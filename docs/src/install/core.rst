****************
Installing Oríon
****************

Oríon is compatible with most Linux distributions and Mac OS X. It is tested on Ubuntu 16.04 LTS and
Mac OS X 10.13. We do not support Windows and there is no short term plan to do so.

The easiest way to install the latest version of Oríon is through the Python package manager. Oríon
is registered on PyPI_ under `orion`. Use the following command to install Oríon:

.. code-block:: sh

   pip install orion

Note that the only algorithm provided by default is random search. More algorithms are available in
the :doc:`plugin section </plugins/install>`, their installation procedure is the same as Oríon's.

.. _PyPI: https://pypi.org/project/orion/

Bleeding edge
=============

If you want to work with the bleeding edge version of Oríon, we recommend you install it with the
following command:

.. code-block:: sh

   pip install git+https://github.com/epistimio/orion.git@develop

Note that the bleeding edge branch is develop. The master branch is the same as the latest version
on PyPI.
