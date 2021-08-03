import math
from typing import Dict, Any

import numpy as np

from orion.benchmark.task.utils import dict_intersection

from .profet.fcnet import FcNetTaskHParams
from .profet.forrester import ForresterTaskHParams
from .profet.svm import SvmTaskHParams
from .profet.xgboost import XgBoostTaskHParams
from .profet.profet_task import spaces

# Since the floats in the 'space strings' generated from the HyperParameters
# class might be string-formatted slightly differently than those specified in
# the global `spaces` dict, we evaluate the expressions and compare the contents
# after they get converted to floats by this mock function.


def mock(min: str, max: str, default_value: Any = None, discrete: bool = False) -> Dict:
    # Mock 'uniform', 'loguniform' function.
    return {
        "min": float(min),
        "max": float(max),
        "discrete": discrete,
        # "default_falue": default_value.
    }


mock_locals = dict(
    loguniform=mock, log_uniform=mock, uniform=mock, np=np, numpy=np, math=math,
)


def assert_orion_space_strings_are_equivalent(space1: Dict, space2: Dict) -> bool:
    assert space1.keys() == space2.keys()
    for n, (v1, v2) in dict_intersection(space1, space2):
        f1: Dict = eval(v1, None, mock_locals)
        f2: Dict = eval(v2, None, mock_locals)
        assert f1 == f2, n


def test_orion_space_strings_match_svm():
    a = spaces["svm"]
    b = SvmTaskHParams.get_orion_space_dict()
    assert_orion_space_strings_are_equivalent(a, b)


def test_orion_space_strings_match_forrester():
    a = spaces["forrester"]
    b = ForresterTaskHParams.get_orion_space_dict()
    assert_orion_space_strings_are_equivalent(a, b)


def test_orion_space_strings_match_fcnet():
    a = spaces["fcnet"]
    b = FcNetTaskHParams.get_orion_space_dict()
    assert_orion_space_strings_are_equivalent(a, b)


def test_orion_space_strings_match_xgboost():
    a = spaces["xgboost"]
    b = XgBoostTaskHParams.get_orion_space_dict()
    assert_orion_space_strings_are_equivalent(a, b)
