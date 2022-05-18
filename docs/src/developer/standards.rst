***********
Conventions
***********

In this chapter, we present the different standards and guidelines we use throughout the project.
All the conventions are enforced automatically during the PR process.

You can verify if your code will pass the checks locally beforehand using ``$ tox -e lint`` (which
is the equivalent of ``$ tox -e black,isort,pylint,doc8,packaging``).

.. _standard-coding:

Coding standard
===============

Our coding standards are specified via black_, isort_ and pylint_. Their configurations are provided
in ``tox.ini`` and ``.pylintrc`` respectively. You can verify the conformity of your changes locally
by running ``$ tox -e black``, ``$ tox -e isort`` and ``$ tox -e pylint``. There is also 2 tox
commands provided to help fix black and isort issues; ``$ tox -e run-black`` and
``$ tox -e run-isort``.

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

We discourage rebases after the pull request has been submitted as it can cause problems in
GitHub's review system which makes it loose track of the comments on the pull request. On another
note, merges are always done with the creation of a merge commit, also known as a *non fast-forward
merge*.

In some cases where the pull request embodies contributions which are scattered across multiple
commits containing incremental changes (e.g., ``fix pep8``, ``update based on feedback``), the pull
request may be integrated to the development branch using `squash and merge <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits>`_
by the maintainer to avoid clutter. It is strongly encouraged to make small pull requests. They are
simpler to implement, easier to integrate and faster to review.

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

GitHub labels
=============

The labels are divided in a few categories.
The objective is to have precise labels while staying lean.
Each category is identified with a color.
Bold colors should be used for tags that should be easily findable when looking at the issues.

Topic
-----

Specifies an area in the software or meta concerns.

* |documentation|_
* |tests|_

.. |documentation| replace:: ``documentation``
.. |tests| replace:: ``tests``
.. _documentation: https://github.com/Epistimio/orion/labels/documentation
.. _tests: https://github.com/Epistimio/orion/labels/tests

Addition
--------

Specifies a new feature.

* |feature|_

.. |feature| replace:: ``feature``
.. _feature: https://github.com/Epistimio/orion/labels/feature

Improvement
-----------

Improves a feature or non-functional aspects (e.g., optimization, prettify, technical debt)

* |enhancement|_

.. |enhancement| replace:: ``enhancement``
.. _enhancement: https://github.com/Epistimio/orion/labels/enhancement

Problems
--------

Indicates an unexpected problem or unintended behavior

* |bug|_

.. |bug| replace:: ``bug``
.. _bug: https://github.com/Epistimio/orion/labels/bug

Status
------

Status of the issue or Priority

* |blocked|_
* |in progress|_
* |in review|_

.. |blocked| replace:: ``blocked``
.. _blocked: https://github.com/Epistimio/orion/labels/blocked
.. |in progress| replace:: ``in progress``
.. _in progress: https://github.com/Epistimio/orion/labels/in%20progress
.. |in review| replace:: ``in review``
.. _in review: https://github.com/Epistimio/orion/labels/in%20review

Discussion
----------

Questions or feedback about the project

* |user question|_
* |dev question|_
* |feedback|_

.. |user question| replace:: ``user question``
.. _user question: https://github.com/Epistimio/orion/labels/user%20question
.. |dev question| replace:: ``dev question``
.. _dev question: https://github.com/Epistimio/orion/labels/dev%20question
.. |feedback| replace:: ``feedback``
.. _feedback: https://github.com/Epistimio/orion/labels/feedback

Community
---------

Related to the community, calls to application

* |help wanted|_
* |good first issue|_

.. |help wanted| replace:: ``help wanted``
.. _help wanted: https://github.com/Epistimio/orion/labels/help%20wanted
.. |good first issue| replace:: ``good first issue``
.. _good first issue: https://github.com/Epistimio/orion/labels/good%20first%20issue

Priority
--------

Qualifies priority bugs and features.
This category enables the maintainers to identify which issues should be done in priority.
Each label has a different shade based on the priority.

* |critical|_
* |high|_
* |medium|_
* |low|_

.. |critical| replace:: ``critical``
.. _critical: https://github.com/Epistimio/orion/labels/critical
.. |high| replace:: ``high``
.. _high: https://github.com/Epistimio/orion/labels/high
.. |medium| replace:: ``medium``
.. _medium: https://github.com/Epistimio/orion/labels/medium
.. |low| replace:: ``low``
.. _low: https://github.com/Epistimio/orion/labels/low

Inactive
--------

No action needed or possible. The issue is either fixed, addressed

* |on hold|_
* |won't fix|_
* |duplicate|_
* |invalid|_

.. |on hold| replace:: ``on hold``
.. _on hold: https://github.com/Epistimio/orion/labels/on%20hold
.. |won't fix| replace:: ``won't fix``
.. _won't fix: https://github.com/Epistimio/orion/labels/wont%20fix
.. |duplicate| replace:: ``duplicate``
.. _duplicate: https://github.com/Epistimio/orion/labels/duplicate
.. |invalid| replace:: ``invalid``
.. _invalid: https://github.com/Epistimio/orion/labels/invalid

.. _Github: https://github.com
.. _doc8: https://pypi.org/project/doc8/
.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pycqa.github.io/isort/
.. _pylint: https://www.pylint.org/
.. _check-manifest: https://pypi.org/project/check-manifest/
.. _readme_renderer: https://pypi.org/project/readme_renderer/
.. _PyPI: https://pypi.org/
