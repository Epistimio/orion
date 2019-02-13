****************************
Installation of Orion's core
****************************

Via PyPI
========

The easiest way to install Oríon is using the Python package manager. The core of Oríon is
registered on PyPI under `orion.core`.

.. code-block:: sh

   pip install orion.core

This will install all the core components. Note that the only algorithm provided with it
is random search. To install more algorithms, you can look at section :doc:`/install/plugins`.

Via Git
=======

This way is recommended if you want to work with the bleeding edge version
of Oríon.

Recommended for users
---------------------

.. code-block:: sh

   pip install git+https://github.com/mila-udem/orion.git@develop

Note that the bleeding-edge branch is develop. The master branch is the same as the latest version
on PyPI.

Recommended for developers of Oríon
-----------------------------------

Clone remote repository_ from Github, using *https* or *ssh*, and then
deploy the project in `development mode`_, by invoking the ``setup.py`` script
with ``develop`` argument
or by using ``pip install --editable``. Usage of a `virtual environment`_ is
also recommended, but not necessary. Example:

.. code-block:: sh

   <Create and navigate to the directory where local repo will reside>
   git clone https://github.com/epistimio/orion.git --branch develop
   mkvirtualenv -a $PWD/orion orion
   python setup.py develop --optimize=1
   <Do your job>
   deactivate

Begin reading instructions for developing it in :doc:`/developer/testing`.

.. _repository: https://github.com/epistimio/orion
.. _virtual environment: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#mkvirtualenv
.. _development mode: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode
