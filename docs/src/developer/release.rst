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
#. Create a new pull request for the branch created in the last step and list all the changes by
   category. Example: https://github.com/Epistimio/orion/pull/283.
#. Update the **Citation** section in the project's README.rst with the latest version of Oríon.
#. Update the ``ROADMAP.md``.
#. Update the linters ``black``, ``isort``, ``pylint``, and ``doc8`` to their latest versions in
   ``tox.ini``, and address any new error.
#. Run the stress tests according to the instructions in stress test's documentation.

.. _release-make:

Making the release
==================
Once the release is thoroughly tested and the core contributors are confident in the release, it's
time to create the release artifacts and publish the release.

#. Merge the release candidate branch to master (no fast-forward merge, we want a merge commit).
#. Create a `new draft release <https://github.com/Epistimio/orion/releases/new>` on GitHub. Set the
   target branch to *master* and the tag version to ``v{version}``. Reuse the changelog from the
   release candidate pull request's for the description. See the `0.1.6
   <https://github.com/Epistimio/orion/releases/tag/v0.1.6>`_ version example.
#. Merge the master branch back to develop.
#. Delete the release candidate branch.
#. Update the backward compability tests by adding the new version in develop branch
   and make a pull request on develop.

Once the release is made, the :ref:`ci` will be automatically started by Github. The code will
then be published on PyPI_ and Anaconda_ automatically if the tests passes.

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

.. _GitHub: https://github.com/Epistimio/orion/releases
.. _Zenodo: https://doi.org/10.5281/zenodo.3478592
.. _PyPI: https://pypi.org/project/orion/
.. _Anaconda: https://anaconda.org/epistimio/orion
