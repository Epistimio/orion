*********
Releasing
*********

In this document, we describe the procedure used to release a new version of Oríon to the public.
Release artifacts are distributed through PyPI_.

Creating a release candidate
============================
The first step in releasing a new version is to create a release candidate. A release candidate
allows us to throughly test the new version and iron out the remaining bugs. Additionally, it's also
at this time that we make sure to change the version number and update related documentation such as
the README.md.

#. Create a new branch from the *develop* branch named ``release-{version}rc``, where
   ``{version}`` is replaced by the number of the new version (e.g., ``1.2.0``). This effectively
   freezes the feature set for this new version, while allowing regular development to continue take
   place in the *develop* branch. More information is available in :ref:`standard-vcs`.``{version}
#. Create a new pull request for the branch created in the last step and list all the changes by
   category. Example: https://github.com/Epistimio/orion/pull/283.
#. Run the stress tests according to the instruct-codingions in stress test's documentation.
#. Update the **Citation** section in the project's README.md with the latest version of Oríon.

.. _release-make:

Making the release
==================
Once the release is throughly tested and the core contributors are confident in the release, it's
time to create the release artifacts and publish the release.

#. Merge the release candidate branch to master (no fast-forward merge, we want a merge commit).
#. Create a `new draft release <https://github.com/Epistimio/orion/releases/new>` on GitHub. Set the
   target branch to *master* and the tag version to ``v{version}``. Reuse the changelog from the
   release candidate pull request's for the descriptione. See the `0.1.6
   <https://github.com/Epistimio/orion/releases/tag/v0.1.6>` version example.
#. Merge the release candidate branch back to develop.
#. Delete the release candidate branch.

Publishing the release
======================
Once the release is correctly documented and integrated to the VCS workflow, we can publish it to
the public.

* Publish the GitHub release. The source code archives will be added automatically by GitHub to the
  release.
* Publish the new version to PyPI_ by executing ``$ tox -e release`` from the tagged commit on the
  master branch.

After the release
=================
Once published, it's important to notify our user base and community that a new version exists so
they can update their version or learn about Oríon.

* Verify Oríon's Zenodo_ page has been updated to reflect the new release on GitHub_. Zenodo is
  configured to automatically create a new version whenever a new release is published on GitHub.
* Announce the new release on your #orion's slack channel.
* Announce the new release on relevant communication channels (e.g., email, forums, google groups)
* Celebrate! You published a new version of Oríon! Congratulations!

.. _GitHub: https://github.com/Epistimio/orion/releases
.. _Zenodo: https://doi.org/10.5281/zenodo.3478592
.. _PyPI: https://pypi.org/project/orion/
