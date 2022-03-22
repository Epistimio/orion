"""Utility to silence loguru"""


def passmethod(*_args, **_kwargs):
    """Method that does nothing"""
    pass


# pylint:disable=too-few-public-methods
class NoLogger:
    """Fake logger to prevent file creation by loguru"""

    def __getattr__(self, _item):
        """always return something that does nothing"""
        return passmethod


def remove_loguru():
    """Override loguru logger class to ignore all its call"""
    # TODO: wtf is that?

    # import loguru

    # Loguru creates 3 log files everytime it is ran
    # We disable it by overriding its definition
    # has to be overriden before DEHB is imported
    # loguru.logger = NoLogger()
