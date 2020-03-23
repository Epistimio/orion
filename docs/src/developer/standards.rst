***********
Conventions
***********

In this chapter, we present the different standards we use throughout the project. The coding and
documentation standards are enforced during the PR process automatically.

You can verify if your code will pass the checks locally beforehand using:

.. code-block:: sh

   tox -e lint

.. _standard-coding:

Coding standard
===============

Our coding standards are specified via flake8_ and pylint_. Their configurations are provided in
``tox.ini`` and ``.pylintrc`` respectively. You can verify the conformity of your changes locally
by running ``$ tox -e flake8`` and ``$ tox -e pylint``.

.. _standard-vcs:

Version Control Guidelines
==========================

To collaborate through VCS, we follow the
`gitflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_
workflow. The *develop* and *master* branches are protected and can only be changed with pull
requests.

For branch names, we recommend prefixing the name of the branch with *feature/*, *fix/*, et
*doc/* depending on the change you're implementing. Additionally, we encourage adding the issue's id
at the start of the branch name, after the prefix. For example the branch name for a bug represented
by issue 225 would be ``fix/225-short-bug-description``.

When creating a release, we use the pattern *release-{version}rc*. This branch represent the release
candidate that will be merge in the master branch when the changes are ready to be launched in
production.

Regarding merges, we recommend you keep your changes in your forked repository for as long as
possible and rebase your branch to Oríon's develop branch before submitting a PR.
Most probably, the develop branch will have changed by the time your PR is approved. In such cases,
we recommend to merge the changes from develop to your branch and then merge your branch to develop.
We discourage rebases after the PR has been submitted as it can cause problems in GitHub's review
system. On another note, merges are always done with the creation of a merge commit, also known
as a *non fast-forward merge*.

.. _standard-documenting:

Documenting standard
====================

Our documentation standard is upheld via doc8_. You can verify your documentation modifications
by running ``$ tox -e doc8``.

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
