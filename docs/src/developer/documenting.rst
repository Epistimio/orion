***********
Documenting
***********
The documentation is built using Sphinx_ with the `Read The Docs`_ theme.

We try to write the documentation at only one place and reuse it as much as possible. For instance,
the home page of this documentation (https://orion.readthedocs.io/) is actually pulled
from the README.rst and appended with a table of content of the documentation generated
automatically. The advantage of having a single source of truth is that it's vastly easier to find
information and keep it up to date.

Updating README.rst
===================

When you need to reference a page from the documentation on the README.rst, make sure to always
point to the **stable** channel in readthedocs (https://orion.readthedocs.io/en/stable/).

If you need to add a link to a specific page in the documentation that is not yet on the stable
channel, make the link to the **latest** channel (https://orion.readthedocs.io/en/latest/). During
the :doc:`release process </developer/release>` the link will be updated to the stable channel.

Building documentation
======================

To generate the *html* and *man* pages of the documentation, run:

.. code-block:: sh

   tox -e docs

When writing, you can run ``$ tox -e serve-docs`` to host the content of
``/docs/build/html`` on http://localhost:8000.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http://www.sphinx-doc.org/en/master/
