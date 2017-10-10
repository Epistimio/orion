from ._version import get_versions
versions = get_versions()
del get_versions

__version__ = versions['version']
