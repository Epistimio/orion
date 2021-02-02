****************
Installing Oríon
****************

Oríon is compatible with most Linux distributions and Mac OS X.
Windows 10 is also supported through the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_.
Oríon is tested on Ubuntu 16.04 LTS and Mac OS X 10.13.

The easiest way to install the latest version of Oríon is through the Python package manager. Oríon
is registered on PyPI_ under ``orion``. Use the following command to install Oríon:

.. code-block:: sh

   pip install orion

Note that Oríon comes with the following algorithms: Random Search, Hyperband, TPE, and ASHA. More
algorithms are available in the :doc:`plugin section </plugins/install>`, their installation is also
done through ``pip``.

Afterwards, we recommend to :doc:`select a database </install/database>` for Oríon to use unless
you're comfortable with the default option.

.. _PyPI: https://pypi.org/project/orion/

Bleeding edge
=============

If you want to work with the bleeding edge version of Oríon, we recommend you install it with the
following command:

.. code-block:: sh

   pip install git+https://github.com/epistimio/orion.git@develop

Note that the bleeding edge branch is develop. The master branch is the same as the latest version
on PyPI.
