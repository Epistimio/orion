"""Example usage and tests for :mod:`orion.core.utils.backward`."""
import pytest

import orion.core.utils.backward as backward
from orion.algo.base import BaseAlgorithm


def create_algo_class(**attributes):
    """Create algo class with given requirements attributes"""
    return type("Algo", (BaseAlgorithm,), attributes)


TYPE_REQUIREMENTS = ["real", "integer", "numerical"]


@pytest.mark.parametrize("requirement", TYPE_REQUIREMENTS)
def test_type_requirements(requirement):
    """Test algorithms type requirement defined in old requirements API"""
    algo_class = create_algo_class(requires=[requirement])
    requirements = backward.get_algo_requirements(algo_class)
    assert len(requirements) == 3
    assert requirements == {
        "type_requirement": requirement,
        "shape_requirement": None,
        "dist_requirement": None,
    }


@pytest.mark.parametrize("requirement", TYPE_REQUIREMENTS)
def test_shape_requirements(requirement):
    """Test algorithms shape requirement defined in old requirements API"""
    algo_class = create_algo_class(requires=[requirement, "flattened"])
    requirements = backward.get_algo_requirements(algo_class)
    assert len(requirements) == 3
    assert requirements == {
        "type_requirement": requirement,
        "shape_requirement": "flattened",
        "dist_requirement": None,
    }


@pytest.mark.parametrize("requirement", TYPE_REQUIREMENTS)
def test_dist_requirements(requirement):
    """Test algorithms dist requirement defined in old requirements API"""
    algo_class = create_algo_class(requires=[requirement, "linear"])
    requirements = backward.get_algo_requirements(algo_class)
    assert len(requirements) == 3
    assert requirements == {
        "type_requirement": requirement,
        "shape_requirement": None,
        "dist_requirement": "linear",
    }


@pytest.mark.parametrize("requirement", TYPE_REQUIREMENTS)
def test_no_changes(requirement):
    """Test algorithms following new requirements API"""
    algo_class = create_algo_class(
        requires_type=requirement, requires_shape="flattened", requires_dist="linear"
    )
    requirements = backward.get_algo_requirements(algo_class)
    assert len(requirements) == 3
    assert requirements == {
        "type_requirement": requirement,
        "shape_requirement": "flattened",
        "dist_requirement": "linear",
    }
