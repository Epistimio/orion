#!/usr/bin/env python
"""Perform a functional test of the debug verbosity level."""
import logging
import os

import pytest

import orion.core.cli


def test_version_print_debug_verbosity(caplog):
    """Tests that Orion version is printed in debug verbosity level"""

    caplog.set_level(logging.DEBUG)

    with pytest.raises(SystemExit):
        orion.core.cli.main([""])
    assert "Orion version : " not in caplog.text

    caplog.clear()
    with pytest.raises(SystemExit):
        orion.core.cli.main(["-vv"])
    for (loggername, loggerlevel, text) in caplog.record_tuples:
        assert not (
            text.startswith("Orion version : ") and (loggerlevel != logging.DEBUG)
        )
    assert "Orion version : " in caplog.text


def test_log_directory(tmp_path):
    """Tests that Orion creates a log directory when the option is specified"""

    logging.shutdown()

    logdir = f"{tmp_path}/mydir"
    with pytest.raises(SystemExit):
        orion.core.cli.main(["-vvv", "--logdir", logdir])

    assert os.path.exists(logdir), "Log dir was created"
