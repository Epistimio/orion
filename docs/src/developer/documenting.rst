***********
Documenting
***********
The documentation is built using Sphinx_ with the `Read The Docs`_ theme.

We try to write the documentation at only one place and reuse it as much as possible. For instance,
the home page of this documentation (https://orion.readthedocs.io/en/latest/) is actually pulled
from the README.md and appended with a table of content of the documentation generated
automatically. The advantage of having a single source of truth is that it's vastly easier to find
information and keep it up to date.

Building documentation
======================

To generate the *html* and *man* pages of the documentation, run:

.. code-block:: sh

   tox -e docs

When writing, you can run ``$ tox -e serve-docs`` to host the content of
``/docs/build/html`` on http://localhost:8000.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http://www.sphinx-doc.org/en/master/
