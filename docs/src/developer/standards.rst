***********
Conventions
***********

In this chapter, we present the different standards and guidelines we use throughout the project.
All the conventions are enforced automatically during the PR process.

You can verify if your code will pass the checks locally beforehand using ``$ tox -e lint`` (which
is the equivalent of ``$ tox -e flake8,pylint,doc8,packaging``).

.. _standard-coding:

Coding standard
===============

Our coding standards are specified via flake8_ and pylint_. Their configurations are provided in
``tox.ini`` and ``.pylintrc`` respectively. You can verify the conformity of your changes locally
by running ``$ tox -e flake8`` and ``$ tox -e pylint``.

In addition, we follow `Numpy's docstring standards
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ to ensure a good
quality of documentation for the project.

.. _standard-vcs:

Version Control Guidelines
==========================

To collaborate through VCS, we follow the
`gitflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_
workflow. The *develop* and *master* branches are protected and can only be changed with pull
requests.

For branch names, we recommend prefixing the name of the branch with *feature/*, *fix/*, and
*doc/* depending on the change you're implementing. Additionally, we encourage adding the issue's id
(if there is one) at the start of the branch name, after the prefix. For example the branch name for
a bug represented by issue 225 would be ``fix/225-short-bug-description``.

When creating a release, we use the pattern *release-{version}rc*. This branch represent the release
candidate that will be merged in the master branch when the changes are ready to be launched in
production.

Synchronization
---------------
Regarding merges, we recommend you keep your changes in your forked repository for as long as
possible and rebase your branch to Oríon's develop branch before submitting a pull request.

Most probably, the develop branch will have changed by the time your pull request is approved. In
such cases, we recommend that you merge the changes from develop to your branch when the reviewer
approves your pull request and then the maintainer will merge your branch to develop, closing your
pull request.

We discourage rebases after the pull resquest has been submitted as it can cause problems in
GitHub's review system which makes it loose track of the comments on the pull request. On another
note, merges are always done with the creation of a merge commit, also known as a *non fast-forward
merge*.

In some cases where the pull request embodies contributions which are scattered across multiple
commits containing incremental changes (e.g., ``fix pep8``, ``update based on feedback``), the pull
request may be integrated to the development branch using `squash and merge <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits>`_
to avoid clutter. It is strongly encouraged to make small pull requests. They are simpler to
implement, easier to integrate and faster to review.

.. _standard-documenting:

Documenting standard
====================

Our documentation standard is upheld via doc8_. You can verify your documentation modifications by
running ``$ tox -e doc8``. The information about writing and generating documentation is available
in the :doc:`documenting` chapter.

.. _standard-repository:

Repository standard
===================

We are using check-manifest_ to ensure no file is missing when we distribute our application and we
also use readme_renderer_ which checks whether ``/README.rst`` can be rendered in PyPI_.
You can verify these locally by running ``$ tox -e packaging``.

Versioning standard
===================

We follow the `semantic versioning <https://semver.org/>`_ convention to name the versions of Oríon.
While in beta, we prepend a ``0`` on the left of the major version.

.. _Github: https://github.com
.. _flake8: http://flake8.pycqa.org/en/latest/
.. _doc8: https://pypi.org/project/doc8/
.. _pylint: https://www.pylint.org/
.. _check-manifest: https://pypi.org/project/check-manifest/
.. _readme_renderer: https://pypi.org/project/readme_renderer/
.. _PyPI: https://pypi.org/
