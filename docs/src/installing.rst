*******
Install
*******

Via PyPI
========

**TBA**
We are soon releasing our first Python distribution on PyPI_.

.. _PyPI: https://pypi.python.org/pypi

Via Git
=======

This way is recommended if you want to work with the bleeding edge version
of Oríon.

User's recommended
------------------

.. code-block:: sh

   pip install git+https://github.com/mila-udem/orion.git@master

Begin reading instructions for using it in :doc:`/user/executing`.

Developer's recommended
-----------------------

Clone remote repository_ from Github, using *https* or *ssh*, and then
deploy the project in `development mode`_, by invoking the ``setup.py`` script
with ``develop`` argument
or by using ``pip install --editable``. Usage of a `virtual environment`_ is
also recommended, but not necessary. Example:

.. code-block:: sh

   <Create and navigate to the directory where local repo will reside>
   git clone https://github.com/mila-udem/orion.git
   mkvirtualenv -a $PWD/orion orion
   workon orion
   python setup.py develop --optimize=1
   <Do your job>
   deactivate

Begin reading instructions for developing it in :doc:`/developer/testing`.

.. _repository: https://github.com/mila-udem/orion
.. _virtual environment: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#mkvirtualenv
.. _development mode: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode
