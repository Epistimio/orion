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

After the release
=================
