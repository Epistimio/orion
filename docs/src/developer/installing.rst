Getting started
===============

In this section, we'll guide you to install the dependencies and environment to develop on Oríon.
We made our best to automate most of the process using Python's ecosystem goodness to facilitate
your onboarding. Let us know how we can improve!

Installing Oríon
----------------
The first step is to clone your remote repository from Github (if not already done, make sure to
fork our repository_ first).

.. code-block:: sh

   git clone https://github.com/epistimio/orion.git --branch develop

.. tip::

   The usage of a `virtual environment`_ is recommended, but not necessary.

   .. code-block:: sh

      mkvirtualenv -a $PWD/orion orion

Then, you need to deploy the project in `development mode`_ by invoking the ``setup.py`` script with
``develop`` argument or by using ``pip install --editable``.

.. code-block:: sh

   python setup.py develop --optimize=1

Setting up the database
-----------------------

Follow the same instructions as to install the :ref:`install_database` regularly.

Verifying the installation
--------------------------

Check everything is ready by running the test suite using ``$ tox`` (this will take some time).
If the tests can't be run to completion, contact us by opening a `new issue <https://github.com/Epistimio/orion/issues/new>`_. We'll do our best to help you!

.. _repository: https://github.com/epistimio/orion
.. _virtual environment: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#mkvirtualenv
.. _development mode: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode
