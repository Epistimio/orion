"""Perform a functional test of the hunt command."""
import pytest

import orion.core.cli


def test_hunt_no_prior(clean_db, one_experiment):
    """Test at least one prior is specified"""
    with(pytest.raises(ValueError)) as exception:
        orion.core.cli.main(["hunt", "-n", "test", "./black_box.py"])

    assert "No prior found" in str(exception.value)
