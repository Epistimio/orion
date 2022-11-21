"""Windows compatibility utilities"""
import os
import time


def getuser():
    """getpass use pwd which is UNIX only"""

    if os.name == "nt":
        return os.getlogin()

    import getpass

    return getpass.getuser()



# pylint: too-few-public-methods
class _readline:
    # pylint: no-method-argument
    def set_completer_delims(*args, **kwargs):
        """Fake method for windows"""


def get_readline():
    """Fake readline interface, readline is UNIX only"""
    if os.name == "nt":
        return _readline

    import readline

    return readline


readline = get_readline()


def replace(old, new, tries=3, sleep=0.01):
    """Windows file replacing is more strict than linux"""
    if os.name != "nt":
        # Rename on UNIX is practically atomic
        # so we use that
        os.rename(old, new)
        return

    # Rename raise an exception on windows if the file exists
    # so we have to use replace
    #
    # if the file is open already windows will raise permission error
    # even if the lock was free, waiting a bit usually fix the issue
    exception = None
    for _ in range(tries):
        try:
            os.replace(old, new)
            return
        except PermissionError as exc:
            time.sleep(sleep)
            exception = exc

    if exception:
        raise exception
