import dataclasses
import itertools
from functools import singledispatch
from inspect import isclass
from logging import getLogger as get_logger
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple, Type, TypeVar, Union

import numpy as np

# from warmstart.translation import SimpleTranslator, Translator
# from .utils import dict_intersection, dict_union, field_dict
from simple_parsing import Serializable
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from simple_parsing.helpers.hparams.priors import (
    LogUniformPrior,
    Prior,
    UniformPrior,
)

logger = get_logger(__name__)
HP = TypeVar("HP")


K = TypeVar("K")
V = TypeVar("V")


def zip_dicts(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[Optional[V], ...]]]:
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key) for d in dicts))


def dict_union(*dicts: Dict) -> Dict:
    """ Simple dict union until we use python 3.9
    
    >>> from collections import OrderedDict
    >>> a = OrderedDict(a=1, b=2, c=3)
    >>> b = OrderedDict(c=5, d=6, e=7)
    >>> dict_union(a, b)
    OrderedDict([('a', 1), ('b', 2), ('c', 5), ('d', 6), ('e', 7)])
    """
    result: Dict = None  # type: ignore
    for d in dicts:
        if result is None:
            result = type(d)()
        result.update(d)
    assert result is not None
    return result


def dict_intersection(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[V, ...]]]:
    common_keys = set(dicts[0])
    for d in dicts:
        common_keys.intersection_update(d)
    for key in common_keys:
        yield (key, tuple(d[key] for d in dicts))


def intersection_over_union(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))


@singledispatch
def distance(a: Any, b: Any, **kwargs) -> float:
    raise NotImplementedError(
        f"No implementation of the distance function for input {a} of type {type(a)}."
    )


@singledispatch
def similarity(a: Any, b: Any, **kwargs) -> float:
    # If there is no registered similarity function, we use an inverse function of the
    # distance.
    d = distance(a, b, **kwargs)
    assert d >= 0, "distance should be non-negative."
    # TODO: Should probably actually use 1 / 2 ** d or a real sigmoid.
    return 1 / (1 + d)


@distance.register
def _distance_between_priors(a: Prior, b: Prior, **kargs) -> float:
    raise NotImplementedError(
        f"No implementation of the distance function for input {a} of type {type(a)}."
    )


@similarity.register
def _similarity_between_uniform_priors(
    a: UniformPrior, b: UniformPrior, **kwargs
) -> float:
    """ Return the 'overlap' between the regions of the two priors.
    """
    # If there is no overlap, return 0.
    if a.min >= b.max or a.max <= b.min:
        return 0.0
    # return the 'intersection over union' of the ranges:
    intersection = min(a.max, b.max) - max(a.min, b.min)
    union = max(a.max, b.max) - min(a.min, b.min)
    return intersection / union


@similarity.register
def _similarity_between_log_uniform_priors(
    a: LogUniformPrior, b: LogUniformPrior, **kwargs
) -> float:
    """ Return the 'overlap' between the regions of the two priors.
    """
    # If there is no overlap, return 0.
    if a.min >= b.max or a.max <= b.min:
        return 0.0
    # return the 'intersection over union' of the ranges, in log-space.:
    intersection = min(a.log_max, b.log_max) - max(a.log_min, b.log_min)
    union = max(a.log_max, b.log_max) - min(a.log_min, b.log_min)
    return intersection / union


def hp_family_distance(
    type_a: Union[Type[HP], HP, Dict],
    type_b: Union[Type[HP], HP, Dict],
    translate: bool = False,
):
    return 1 - hp_family_similarity(type_a, type_b, translate=translate)


@similarity.register(type)
def hp_family_similarity(type_a: Type[HP], type_b: Type[HP], translate: bool = False):
    # Crude estimate of the similarity between two families of HParams:
    # Use the intersection over the union of the keys of the dictionaries, multiplied
    # by the intersection over union of the priors for common spaces.
    assert issubclass(type_a, HyperParameters)
    assert issubclass(type_b, HyperParameters)

    # The value to be returned. Each factor will be multiplicatively added to this.
    space_similarity = 1.0

    # TODO: Maybe add 'coefficients'/'factors' for each part of the similarity.
    key_similarity_factor: float = 1.0
    space_similarity_factor: float = 1.0

    space_a: Dict[str, Prior] = type_a.get_priors()
    space_b: Dict[str, Prior] = type_b.get_priors()

    if translate:
        # Convert equivalent keys in the search spaces.
        from warmstart.translation import SimpleTranslator

        a_translator = SimpleTranslator.get_translator(type_a)
        space_b = a_translator.translate(space_b)

    # TODO: Should this handle nested dicts?
    keys_similarity = intersection_over_union(space_a, space_b)
    logger.debug(f"{keys_similarity=}")
    # Add this component to the result.
    # NOTE: taking the similarity to a power, since they are all [0, 1]
    space_similarity *= keys_similarity ** key_similarity_factor

    common_spaces = dict(dict_intersection(space_a, space_b))
    # Dict containing the similarities for all the common spaces
    # common_spaces_similarities: Dict[str, float] = {}
    # find the overlap between the priors:
    for key, (value_a, value_b) in common_spaces.items():
        sim = similarity(value_a, value_b)
        logger.debug(f"{key=}, {value_a=}, {value_b=}, {sim=}")
        # common_spaces_similarities[key] = sim
        # Add this component to the result.
        space_similarity *= sim ** space_similarity_factor
    return space_similarity


@distance.register(HyperParameters)
def hp_distance(
    hp1: HyperParameters,
    hp2: HyperParameters,
    weights: Dict[str, float] = None,
    translate: bool = False,
) -> float:
    x1: Dict[str, float] = hp1.to_dict()

    if weights is None:
        weights = {}
        for key, prior in hp1.get_priors().items():
            if not isinstance(prior, UniformPrior):
                raise NotImplementedError(f"TODO: add support for prior {prior}")
            # TODO: Do this differently for LogUniform priors, right?
            length = prior.max - prior.min
            if length == 0 or np.isposinf(length):
                # TODO
                raise NotImplementedError(
                    f"Domain of hparam {key} is infinite: {prior}"
                )
            weights[key] = 1 / length

    if weights:
        if not translate:
            assert set(weights) <= set(
                x1
            ), f"When given, the weights should match some of the fields of {x1} ({weights})"
    else:
        weights = {}
    assert isinstance(hp2, HyperParameters), "Simplifying the code a bit."

    # wether hp1 and hp2 are related through inheritance
    related = isinstance(hp2, type(hp1)) or isinstance(hp1, type(hp2))
    if related:
        # Easiest case: the two are of the same type or related through
        # inheritance, no need to translate anything.
        x2: Dict[str, float] = hp2.to_dict()
    elif translate:
        translator = SimpleTranslator.get_translator(type(hp1))
        logger.debug(f"x2 before: {hp2}")
        # 'Translate' the hp2 into a dict with the same keys as 'hp1'
        x2, _ = translator.translate2(hp2)
        logger.debug(f"x2 after: {x2}")
        if weights:
            logger.debug(f"weights before: {weights}")
            weights, _ = translator.translate2(weights)
            logger.debug(f"Weights after: {weights}")

    else:
        assert isinstance(hp2, Serializable)
        x2 = hp2.to_dict()

    distance: float = 0.0
    x1 = {k: v for k, v in x1.items() if k in weights}
    x2 = {k: v for k, v in x2.items() if k in weights}
    assert weights is not None
    for k, (v1, v2) in dict_intersection(x1, x2):
        distance += weights.get(k, 1) * abs(v1 - v2)
    return distance


@distance.register(dict)
def dict_distance(
    d1: Dict, d2: Dict, weights: Dict[str, float] = None, translate: bool = False
) -> float:
    """ TODO: unused for now, but could use something like this later down the
    line when using dictionaries and no dataclasses (perhaps for Orion integration)
    """
    if weights:
        assert set(weights) <= set(
            d1
        ), f"weights, if given, should match some keys from {d1}"
    else:
        weights = {}

    if set(d1) != set(d2) and translate:
        from warmstart.translation import Translator

        translator: Translator[Dict, Dict] = SimpleTranslator.get_translator(set(d1))
        d2 = translator.translate(d2, drop_rest=True)

    distance = 0.0
    for k, (v1, v2) in dict_intersection(d1, d2):
        if not isinstance(v1, (int, float)):
            raise NotImplementedError(
                "We sortof need the values in the dicts to be ints or floats for now."
            )
        distance += weights.get(k, 1) * abs(v1 - v2)
    return distance


def to_dict(hp: Union[HP, Type[HP], Dict, Serializable]) -> Dict:
    if isinstance(hp, Serializable):
        return hp.to_dict()
    elif dataclasses.is_dataclass(hp):
        if isclass(hp):
            return field_dict(hp)
        return dataclasses.asdict(hp)
    elif isinstance(hp, dict):
        return hp
    else:
        try:
            return dict(hp)
        except Exception as exc:
            raise RuntimeError(
                f"Can't convert {hp} of type ({type(hp)} to a dict! ({exc})"
            )


@similarity.register(tuple)
def point_similarity(
    point_a: Tuple[HyperParameters, float],
    point_b: Tuple[HyperParameters, float],
    perf_dist_coef: float = 1.0,
    translate: bool = True,
) -> float:
    hp_a, perf_a = point_a
    hp_b, perf_b = point_b

    # Value to be returned.
    overall_similarity: float = 1.0

    assert isinstance(hp_a, HyperParameters)
    assert isinstance(hp_b, HyperParameters)

    type_a = type(hp_a)
    type_b = type(hp_b)
    dict_a = hp_a.to_dict()
    dict_b = hp_b.to_dict()

    logger.debug(f"dict a: {dict_a}, perf a: {perf_a}")
    logger.debug(f"dict b: {dict_b}, perf b: {perf_b}")

    # logger.debug(f"extra items: {extra_items}")
    type_similarity = hp_family_similarity(type_a, type_b, translate=translate)
    logger.debug(f"Similarity (families): {type_similarity}")
    overall_similarity *= type_similarity

    # hp_similarity == 0: nothing in common.
    # hp_similarity == 1: same point.
    # by this point if necessary.
    hp_similarity = similarity(hp_a, hp_b, translate=translate)
    logger.debug(f"Similarity (attributes): {hp_similarity}")
    overall_similarity *= hp_similarity

    # TODO: Does this make sense?
    performance_distance = abs(perf_a - perf_b)
    performance_similarity = 1 / (1 + perf_dist_coef * performance_distance)
    logger.debug(f"{performance_distance=}, {perf_dist_coef=}")
    logger.debug(f"Similarity (performance): {performance_similarity}")
    overall_similarity *= performance_similarity

    logger.debug(f"Similarity (overall): {overall_similarity}")
    return overall_similarity
