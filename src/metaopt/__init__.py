from ._version import get_versions
versions = get_versions()
del get_versions

#  if versions['error']:
#      raise RuntimeError(versions['error'])

__version__ = versions['version']

print(versions)
print()
print(__version__)
