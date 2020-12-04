*********
Releasing
*********

In this document, we describe the procedure used to release a new version of Oríon to the public.
Release artifacts are distributed through PyPI_.

Creating a release candidate
============================
The first step in releasing a new version is to create a release candidate. A release candidate
allows us to thoroughly test the new version and iron out the remaining bugs. Additionally, it's
also at this time that we make sure to change the version number and update related documentation
such as the README.rst.

#. Create a new branch from the *develop* branch named ``release-{version}rc``, where
   ``{version}`` is replaced by the number of the new version (e.g., ``1.2.0``). This effectively
   freezes the feature set for this new version, while allowing regular development to continue take
   place in the *develop* branch. More information is available in :ref:`standard-vcs`.
#. In README.rst, replace any link pointing to ``https://orion.readthedocs.io/en/latest/**`` to
   ``https://orion.readthedocs.io/en/stable/**``.
#. Update the **Citation** section in the project's README.rst with the latest version of Oríon.
#. Update the ``ROADMAP.md``.
#. Update the linters ``black``, ``isort``, ``pylint``, and ``doc8`` to their latest versions in
   ``tox.ini``, and address any new error.
#. Run the stress tests according to the instructions in stress test's documentation.
#. Create a new pull request for the branch created in the first step. The pull request should be
   using the base `master` instead of `develop`.
#. Go to the `release page`_ copy paste the current draft
   (which was automatically wrote by the `release drafter`_ app) into the description of the new
   pull request. Adapt if necessary, sometimes new features are spread across multiple pull requests
   or some pull requests are changes that should not figure in the release description, like
   merging master back to develop branch (ex: `PR #510 <https://github.com/Epistimio/orion/pull/510>`_).

.. _release-make:

Making the release
==================
Once the release is thoroughly tested and the core contributors are confident in the release, it's
time to create the release artifacts and publish the release.

#. Merge the release candidate branch to master (no fast-forward merge, we want a merge commit).
#. Delete the release candidate branch.
#. Go back to the `release page`_, edit the title to describe important changes and publish the
   release. The version should already be updated automatically by `release drafter`_. Adjust if
   necessary.
#. The publication should trigger a github action which will update the backward compatibility tests
   and create a pull request to merge the master back to develop branch. Wait for tests to pass and
   merge.

Once the release is made, the :ref:`ci` will be automatically started by Github. The code will
then be published on PyPI_ and Anaconda_ automatically if the tests passes. If test fails for
random reasons (sometimes mongodb setup fails during build), you can fetch the tagged version
locally and publish on PyPI_ and Anaconda_ manually, using ``$ tox -e release`` and
``./conda/upload.sh``.

After the release
=================
Once published, it's important to notify our user base and community that a new
version exists so they can update their version or learn about Oríon.

* Verify Oríon's Zenodo_ page has been updated to reflect the new release on GitHub_. Zenodo is
  configured to automatically create a new version whenever a new release is published on GitHub.
* Verify Oríon's PyPI_ and Anaconda_ page contains the new version. Binaries for the new version are
  uploaded automatically by Github's workflow when the tests pass for the merge commit tagged with
  the new version on the master branch .
* Announce the new release on your #orion's slack channel.
* Announce the new release on relevant communication channels (e.g., email, forums, google groups)
* Congratulations, you published a new version of Oríon!

.. _release drafter: https://github.com/marketplace/actions/release-drafter
.. _release page: https://github.com/Epistimio/orion/releases
.. _GitHub: https://github.com/Epistimio/orion/releases
.. _Zenodo: https://doi.org/10.5281/zenodo.3478592
.. _PyPI: https://pypi.org/project/orion/
.. _Anaconda: https://anaconda.org/epistimio/orion
