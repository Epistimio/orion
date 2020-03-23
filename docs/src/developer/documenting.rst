***********
Documenting
***********
The documentation is built using Sphinx_ with the `Read The Docs`_ theme.

To generate the *html* and *man* pages of the documentation, run:

.. code-block:: sh

   tox -e docs

When writing, you can also run ``$ tox -e serve-docs`` to host the content of
``/docs/build/html`` on http://localhost:8000.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http://www.sphinx-doc.org/en/master/
