"""
Utility functions for importing modules
=======================================

Conversion functions between various data types used in framework's ecosystem.

"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_GIT_EXTRAS_PATH = Path(__file__).parents[2] / "git_extras.toml"


def load_modules_in_path(path, filter_function=None):
    """
    Load all modules inside `path` and return a list of those
    fitting the filter function.
    """
    this_module = __import__(path, fromlist=[""])
    file_path = this_module.__path__[0]

    files = list(
        map(
            lambda f: f.split(".")[0],
            filter(lambda f2: f2.endswith("py"), os.listdir(file_path)),
        )
    )

    modules = map(lambda f: __import__(path + "." + f, fromlist=[""]), files)

    if filter_function is not None:
        modules = filter(filter_function, modules)

    return list(modules)


class ImportOptional:
    """Context manager to handle optional dependencies

    It will catch any import errors occurring inside the with-clause and
    keep a reference to the error so it can be reraised latter on.

    Use import_optional.ensure() to verify the extra dependency is correctly installed,
    otherwise it will raise the previously caught ImportError.

    Parameters
    ----------
    package: str
        Name of the package (ex: algorithm or executor)
    extra_dependency: str, optional
        Name of the extra dependency to be installed.
        Will be used in the error string to guide installation of the
        extra package as ``pip install orion[extra_dependency]``.
        If not defined, will be ``package.lower()``
    """

    def __init__(self, package: str, extra_dependency: str = None) -> None:
        self.package = package
        if extra_dependency is None:
            extra_dependency = package.lower()
        self.extra_dependency = extra_dependency

        self.import_error: None | ImportError = None
        self._git_extras = self._load_git_extras()

    def __enter__(self) -> ImportOptional:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if isinstance(exc_value, ImportError):
            self.import_error = exc_value
            return True

        return False

    @property
    def failed(self) -> bool:
        """Whether the imports during the with-clause all worked out."""
        return bool(self.import_error)

    @staticmethod
    def _load_git_extras():
        """Load the set of extra names defined in git_extras.toml."""
        try:
            with open(_GIT_EXTRAS_PATH, "rb") as f:
                return set(tomllib.load(f).keys())
        except FileNotFoundError:
            return set()

    def ensure(self) -> None:
        """Raise the previously caught ImportError if the import failed."""
        if self.failed:
            if self.extra_dependency in self._git_extras:
                install_cmd = f"orion install {self.extra_dependency}"
            else:
                install_cmd = f"pip install orion[{self.extra_dependency}]"
            raise ImportError(
                f"The package `{self.package}` is not installed. Install it with "
                f"`{install_cmd}`."
            ) from self.import_error
