***************
Getting started
***************
In this section, we'll guide you to install the dependencies and environment to develop on Oríon.
We made our best to automate most of the process using Python's ecosystem goodness to facilitate
your onboarding. Let us know how we can improve!

Oríon
=====
The first step is to clone your remote repository from Github (if not already done, make sure to
fork our repository_ first).

.. code-block:: sh

   $ git clone https://github.com/epistimio/orion.git

.. tip::

   The usage of a `virtual environment`_ is recommended, but not necessary.

   .. code-block:: sh

      $ mkvirtualenv -a $PWD/orion orion

Then, install the project in editable mode with the test extras:

.. code-block:: sh

   $ pip install -e ".[test]"

Database
========
Follow the same instructions as to install the :ref:`install_database` locally.

Verifying the installation
==========================
For developer's convenience the packages enlisted in the requirements file
``dev-requirements.txt`` are meant to facilitate the development process.
Packages include `tox <https://tox.readthedocs.io/en/latest/>`_ for defining
and organizing macros of sh commands in virtual environments, and packages
for linting as we will see in a next chapter.

Check everything is ready by running the test suite using ``$ tox -e py`` (this will
take some time). If the tests can't be run to completion, contact us by opening a `new issue
<https://github.com/Epistimio/orion/issues/new>`_. We'll do our best to help you!

About tox
=========
tox_ is an automation tool that execute tasks in virtual environments. We automate all our testing,
verification, and release macros with it. All contexts are defined in
`tox.ini <https://github.com/epistimio/orion/blob/master/tox.ini>`_. They can be executed using
``$ tox -e <context name>``.

.. _tox: https://tox.readthedocs.io/en/latest/
.. _repository: https://github.com/epistimio/orion
.. _virtual environment: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#mkvirtualenv
