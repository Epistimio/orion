# -*- coding: utf-8 -*-
"""
ContextManager for working directory
====================================

ContextManager class to create a permanent directory or a temporary one.

"""
import os
import tempfile


# pylint: disable=too-few-public-methods
class WorkingDir:
    """ContextManager class for temporary or permanent directory."""

    def __init__(self, working_dir, temp=True, suffix=None, prefix=None):
        """Create the context manager with the given name.

        Parameters
        ----------
        name : str, optional
            Name of the directory. If empty, will create a temporary one.

        """
        self.working_dir = str(working_dir)
        self._temp = temp
        self._suffix = suffix
        self._prefix = prefix
        self._tmpdir = None

    def __enter__(self):
        """Create the a permanent directory or a temporary one."""
        os.makedirs(self.working_dir, exist_ok=True)

        if not self._temp:
            path = os.path.join(self.working_dir, self._prefix + self._suffix)
            os.makedirs(path, exist_ok=True)
            return path

        self._tmpdir = tempfile.TemporaryDirectory(
            suffix=self._suffix, prefix=self._prefix, dir=self.working_dir
        )
        return self._tmpdir.name

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup temporary directory."""
        if self._temp:
            self._tmpdir.cleanup()
