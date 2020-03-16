.. contents:: Developer's Guide 102: Documenting

********
Document
********

We are using `Read The Docs`_ theme for Sphinx_.

Run tox command::

   tox -e docs

to build *html* and *man* pages for documentation

Also by executing::

   tox -e serve-docs

the page under ``/docs/build/html`` is hosted by *localhost*.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http://www.sphinx-doc.org/en/master/
