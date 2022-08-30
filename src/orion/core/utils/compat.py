"""Windows compatibility utilities"""
import os


def getuser():
    """getpass use pwd which is UNIX only"""

    if os.name == 'nt':
        return os.getlogin()

    import getpass
    return getpass.getuser()


class _readline:
    def set_completer_delims(*args, **kwargs):
        """Fake method for windows"""
        pass


def get_readline():
    """Fake readline interface, readline is UNIX only"""
    if os.name == 'nt':
        return _readline

    import readline
    return readline


readline = get_readline()
