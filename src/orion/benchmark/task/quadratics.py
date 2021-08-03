""" Defines the Quadratics task, as described in section 4.2 of the ABLR paper.

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
"""
from typing import ClassVar, Type, Union, List, Dict, Any, Tuple, Dict
import numpy as np
import copy
import math
from dataclasses import dataclass, fields, replace
from warmstart.hyperparameters import HyperParameters, uniform
from warmstart.utils import compute_identity
from warmstart.tasks import Task
from warmstart.distance import similarity, distance

from warmstart.utils import compute_identity
from logging import getLogger as get_logger
from dataclasses import is_dataclass, asdict

logger = get_logger(__name__)


@dataclass
class QuadraticsTaskHParams(HyperParameters):
    # TODO: In the paper, section 4.2, the function is defined over R^3, but I'm
    # limiting the bounds to this for now since the y can be enormous otherwise.
    x0: float = uniform(-100., 100.)
    x1: float = uniform(-100., 100.)
    x2: float = uniform(-100., 100.)


from dataclasses import InitVar
from typing import Optional
from dataclasses import field


@dataclass
class QuadraticsTask(Task, HyperParameters):
    hparams: ClassVar[Type[HyperParameters]] = QuadraticsTaskHParams

    a2: Optional[float] = uniform(0.1, 10.0, default=None)
    a1: Optional[float] = uniform(0.1, 10.0, default=None)
    a0: Optional[float] = uniform(0.1, 10.0, default=None)

    task_id: int = 0
    rng: Optional[np.random.Generator] = field(default=None, repr=False, compare=False)
    seed: Optional[int] = None
    max_trials: int = 50

    def __post_init__(self):
        # super().__post_init__()
        super().__init__(
            task_id=self.task_id,
            rng=self.rng,
            seed=self.seed,
            max_trials=self.max_trials,
        )
        # Note: rounding to 4 decimals just to prevent some potential issues
        # related to the string representation of the task later on.
        self.a2 = np.round(
            self.rng.uniform(0.1, 10.0) if self.a2 is None else self.a2, 4
        )
        self.a1 = np.round(
            self.rng.uniform(0.1, 10.0) if self.a1 is None else self.a1, 4
        )
        self.a0 = np.round(
            self.rng.uniform(0.1, 10.0) if self.a0 is None else self.a0, 4
        )

    def with_context(self) -> "QuadraticsTaskWithContext":
        return QuadraticsTaskWithContext(
            a2=self.a2, a1=self.a1, a0=self.a0, task_id=self.task_id, rng=self.rng,
        )

    def __repr__(self) -> str:
        return f"QuadraticsTask(a0={self.a0}, a1={self.a1}, a2={self.a2})"

    def __call__(
        self,
        hp: Union[
            QuadraticsTaskHParams, List[QuadraticsTaskHParams], np.ndarray
        ] = None,
        **kwargs,
    ) -> np.ndarray:
        if hp is None and kwargs:
            hp = self.hparams(**kwargs)
        if (isinstance(hp, list) and hp and not isinstance(hp[0], float)) or (
            isinstance(hp, np.ndarray) and hp.ndim == 2
        ):
            # This is being called with a batched input.
            return np.array(list(map(self, hp)))
        x: np.ndarray
        if isinstance(hp, np.ndarray):
            x = hp
        elif isinstance(hp, QuadraticsTaskHParams):
            x = hp.to_array()
        else:
            x = np.asfarray(hp)

        if x.shape[-1] > 3:
            # Discard any 'extra' values.
            # NOTE: This is only really used in the 'with context' version of
            # this task below.
            x = x[..., :3]

        assert x.shape == (3,)
        y = 0.5 * self.a2 * (x ** 2).sum() + self.a1 * x.sum() + self.a0

        # Interesting: This polynomial has the same result, when passed x.sum(-1) rather
        # than `x`.
        # from numpy.polynomial import Polynomial
        # poly = Polynomial(
        #     coef=[self.a0, self.a1, self.a2],
        #     domain=[0.1, 10.0],
        #     window=[0.1, 10.0],
        # )
        # y_poly = poly(x.sum(-1))
        # assert False, (y, y_poly)

        return y

    def gradient(self, x: np.ndarray) -> float:
        # if x.ndim == 2:
        #     assert x.shape[-1] == 3
        #     return self.a2 * x.sum(-1) + self.a1
        assert x.shape[-1] == 3
        return self.a2 * x.sum(-1) + self.a1

    @property
    def hash(self) -> str:
        # TODO: Return a unique "hash"/id/key for this task

        if is_dataclass(self):
            return compute_identity(**asdict(self))
        return compute_identity(
            **self.space, a0=self.a0, a1=self.a1, a2=self.a2, task_id=self.task_id
        )

    def __neg__(self) -> "QuadraticsTask":
        return replace(a2=-self.a2, a1=-self.a1, a0=-self.a0)

    def __add__(self, other: Union["QuadraticsTask", Any]) -> "QuadraticsTask":
        if not isinstance(other, type(self)):
            return NotImplemented
        return replace(
            a2=self.a2 + other.a2, a1=self.a1 + other.a1, a0=self.a0 + other.a0
        )

    def __sub__(self, other: Union["QuadraticsTask", Any]) -> "QuadraticsTask":
        if not isinstance(other, type(self)):
            return NotImplemented
        return self + (-other)

    @classmethod
    def get_bounds_dict(cls) -> Dict[str, Tuple[float, float]]:
        """Returns the bounds of the search domain for this type of HParam.

        Returns them as a list of `BoundInfo` objects, in the format expected by
        GPyOpt.
        """
        bounds: Dict[str, Tuple[float, float]] = {}
        for f in fields(cls):
            # TODO: handle a hparam which is categorical (i.e. choices)
            min_v = f.metadata.get("min")
            max_v = f.metadata.get("max")
            if min_v is None or max_v is None:
                continue
            bounds[f.name] = (min_v, max_v)
        return bounds

    @classmethod
    def task_low(cls) -> "QuadraticsTask":
        """ WIP: Get the task at the 'minimum' edge of the task's domain. """
        return cls(**{k: v[0] for k, v in cls.get_bounds_dict().items()})

    @classmethod
    def task_high(cls) -> "QuadraticsTask":
        """ WIP: Get the task at the 'maximum' edge of the task's domain. """
        return cls(**{k: v[1] for k, v in cls.get_bounds_dict().items()})

    def get_similar_task(
        self, correlation_coefficient: float, **kwargs
    ) -> "QuadraticsTask":
        """ WIP: Return a new task, 'correlated' with `self` by a given ratio.

        NOTE: At the moment the returned task will have the same max_trials, search
        space and seed as `self`. The only parameters that currently change are the
        coefficients of the quadratic.

        TODO: (@lebrice) For now, I'm simplifying things a bit by assuming that:

        ```
        similarity = 1 / distance(task_a, task_b) if task_a != task_b else 1.0
        ```
        
        where
        
        ```
        distance(task_a, task_b) = (
            abs(task_a.a2 ** 2 - task_b.a2 ** 2)
            + abs(task_a.a1 - task_b.a1)
            + abs(task_a.a0 - task_b.a0)
        )
        
        ```
        Therefore, we can change the coefficients of the other task, given the
        coefficients of the current task and the desired "correlation coefficient":
        Then, we distribute the "distance to add" over the three possible terms.
        
        Prioritizes changing a1, then a2, then a0, as needed.
        """
        if not (0 <= correlation_coefficient <= 1.0):
            raise RuntimeError(
                f"Correlation 'coefficient' must be between 0 and 1 (got "
                f"{correlation_coefficient})."
            )

        if correlation_coefficient == 1.0:
            identical_task = replace(self, **kwargs)
            return identical_task

        task_low = QuadraticsTask.task_low()
        task_high = QuadraticsTask.task_high()
        if correlation_coefficient == 0.0:
            # Return the "furthest" task between the 'low' and the 'high' tasks.
            furthest_task = max(task_low, task_high, key=self.distance_to)
            kwargs.setdefault("seed", self.seed)
            furthest_task_with_given_attributes = replace(furthest_task, **kwargs)
            return furthest_task_with_given_attributes

        # if self.a2 == 0:
        # Special case, if we're given a task with

        # Here for this particular task we know the most uncorrelated coefficients
        # possible, so we calculate the maximum 'distance' you could possibly put
        # between `self` and another task of the same type:
        max_dist = max(distance(self, task_low), distance(self, task_high))
        assert max_dist > 0
        # The total 'distance' / delta in the parameters that we have to spread around.
        distance_left_to_add = (1 - correlation_coefficient) * max_dist

        logger.debug(f"{distance_left_to_add=} {max_dist=}")
        new_a2: float = self.a2
        new_a1: float = self.a1
        new_a0: float = self.a0

        def vary_value(
            current_value: float, v_min: float, v_max: float, max_abs_delta: float
        ) -> Tuple[float, float]:
            """ Propose a new value, returning the new proposed value and the change.
            """
            assert max_abs_delta >= 0
            assert v_min <= current_value <= v_max, (
                current_value,
                v_min,
                v_max,
                max_abs_delta,
            )

            if max_abs_delta == 0:
                return current_value, 0.0
            room_left_below = current_value - v_min
            room_left_above = v_max - current_value
            if room_left_above > room_left_below:
                assert room_left_above > 0
                delta = min(room_left_above, max_abs_delta)
                new_value = current_value + delta
                if np.isclose(new_value, v_max):
                    new_value = v_max
                assert v_min <= new_value <= v_max
                return new_value, +delta
            else:
                assert room_left_below > 0
                delta = min(room_left_below, max_abs_delta)
                new_value = current_value - delta
                if np.isclose(new_value, v_min):
                    new_value = v_min
                assert v_min <= new_value <= v_max
                return new_value, -delta

        assert distance_left_to_add > 0
        logger.debug(
            f"{correlation_coefficient=}, {max_dist=}, {distance_left_to_add=}, "
        )
        # Prioritize varying the 'a1' coefficient
        new_a1, delta = vary_value(
            current_value=new_a1,
            v_min=0.1,
            v_max=10.0,
            max_abs_delta=distance_left_to_add,
        )
        distance_consumed = abs(delta)
        distance_left_to_add -= distance_consumed

        if distance_left_to_add > 0:
            logger.debug(
                f"Used up all the changes possible in a1, {distance_left_to_add=}"
            )
            # Next, prioritize varying the 'a2' coefficient.
            new_a2, delta = vary_value(
                current_value=new_a2,
                v_min=0.1,
                v_max=10.0,
                max_abs_delta=math.sqrt(distance_left_to_add),
            )
            distance_consumed = delta ** 2
            distance_left_to_add -= distance_consumed

        if distance_left_to_add > 0:
            logger.debug(
                f"Used up all the changes possible in a1 and a2, {distance_left_to_add=}"
            )
            # and lastly, the a0 coefficient.
            new_a0, delta = vary_value(
                current_value=new_a0,
                v_min=0.1,
                v_max=10.0,
                max_abs_delta=distance_left_to_add,
            )
            distance_consumed = abs(delta)
            distance_left_to_add -= distance_consumed

        logger.debug(f"{new_a2=}, {new_a1=}, {new_a0=}, {distance_left_to_add=}")
        # We should have exhausted all the 'budget' we had by the time we get here.
        assert np.isclose(distance_left_to_add, 0)

        kwargs = kwargs.copy()
        kwargs.update(a2=new_a2, a1=new_a1, a0=new_a0)
        new = replace(self, **kwargs)
        return new
        # if self.a2 == 0:
        #     # line-correlation:
        #     # Using http://atozmath.com/example/CONM/Ch4_RegreLines.aspx?he=e&q=3
        #     # Seems like it's something like:
        #     # y1 = m1 * x + b1
        #     # y2 = m2 * x + b2
        #     r = correlation_coefficient
        #     dy_dx = -self.a2

        #     byx = r * dy_dx

        #     r = np.sqrt(dy_dx * dx_dy)

        #     correlation_coefficient

        # r, p_value = pearsonr()


from scipy.stats import pearsonr, spearmanr


@similarity.register
def similarity_between_quadratic_tasks(
    task_a: QuadraticsTask, task_b: QuadraticsTask, n: int = 100, **unused_kwargs
) -> float:
    # difference_task = task_a - task_b
    # unit_input = np.ones(3)
    # difference_output = abs(difference_task(unit_input))
    # x_a, y_a = task_a.make_dataset_np(n)
    # y_b = task_b(x_a)

    # from scipy.stats import pearsonr, spearmanr
    # x_a, y_a = task_a.make_dataset_np(n)
    # y_b = task_b.call(x_a)

    # y_high = task_a.task_high()(x_a)
    # y_low = task_a.task_low()(x_a)

    # r_score, p_value = pearsonr(y_low, y_high)
    # logger.debug(f"Pearson: {r_score=}, {p_value=}")
    # # > (1.0, 0.0)
    # r_score, p_value = spearmanr(y_low, y_high)
    # logger.debug(f"Spearman: {r_score=}, {p_value=}")

    d = distance(task_a, task_b)
    # Here for this particular task we know the most different values possible.
    max_dist = max(
        distance(task_a, QuadraticsTask.task_low()),
        distance(task_a, QuadraticsTask.task_high()),
        distance(task_b, QuadraticsTask.task_low()),
        distance(task_b, QuadraticsTask.task_high()),
    )
    assert 0 <= d <= max_dist
    logger.debug(f"{task_a=} {task_b=}, {d=}, {max_dist=}")
    return 1 - d / max_dist
    # return 1 / d if d != 0 else 1.0
    # return 1 / d

    # spearman_r, spearman_p = spearmanr(y_a, y_b)
    # pearson_r, pearson_p = pearsonr(y_a, y_b)
    # assert False, (spearman_r, spearman_p, pearson_r, pearson_p)
    # return r

    # max_distance = 0
    # y_max = QuadraticsTask.task_high()(x_a)
    # y_min = QuadraticsTask.task_low()(x_a)
    # max_distance = np.abs(y_max - y_min).sum() / n
    # distance = np.abs(y_a - y_b).sum() / n

    # similarity = distance / max_distance
    # return similarity


@distance.register
def distance_between_quadratics_task(
    task_a: QuadraticsTask, task_b: QuadraticsTask, n: int = 100, **unused_kwargs
) -> float:
    """ TODO: 'distance' between two quadratic functions.

    For now using something simple
    """
    distance = (
        abs(task_a.a2 ** 2 - task_b.a2 ** 2)
        + abs(task_a.a1 - task_b.a1)
        + abs(task_a.a0 - task_b.a0)
    )
    return distance


# @similarity.register
# def similarity(
#     task_a: QuadraticsTask, task_b: QuadraticsTask, n: int = 100, **unused_kwargs
# ) -> float:
#     xs = task_a.sample_np(n=n)
#     y_a: np.ndarray = task_a.call(xs)
#     y_b: np.ndarray = task_b.call(xs)
#     # logger.debug(y_a, y_b)
#     # y_a -= y_a.mean()
#     # y_a /= y_a.std()

#     # y_b -= y_b.mean()
#     # y_b /= y_b.std()

#     if (y_a == y_b).all():
#         return 1.0

#     from scipy.stats import pearsonr, wilcoxon
#     from warmstart.utils.utils import sigmoid

#     distance = np.abs(y_a - y_b).sum() / n
#     similarity = 1 / (1 + distance)
#     ret
#     # x_mean = xs.mean(0)
#     # distance_2 = (task_a.a2 - task_b.a2) * (x_mean ** 2) + abs(task_a.a1 - task_b.a1) * x_mean + abs(task_a.a0 - task_b.a0)
#     # assert False, (distance, distance_2)
#     wilcoxon_score, wilcoxon_p_value = wilcoxon(y_a, y_b)  # zero_method="zsplit")
#     pearson_r_score, pearson_p_value = pearsonr(y_a, y_b)
#     assert False, (similarity, wilcoxon_p_value, wilcoxon_score / n, pearson_r_score, pearson_p_value)
#     return r_score


@dataclass
class QuadraticsTaskWithContextHparams(QuadraticsTaskHParams):
    # This adds these entries to the space, but they aren't used in the
    # quadratic equation above. (`__call__` of QuadraticsTask.)
    a2: float = uniform(0.1, 10.0)
    a1: float = uniform(0.1, 10.0)
    a0: float = uniform(0.1, 10.0)


class QuadraticsTaskWithContext(QuadraticsTask):
    """ Same as the QuadraticsTask, but the samples also have the "context" 
    information added to them (a2, a1, a0).
    
    This is used to help demonstrate the effectiveness of multi-task models,
    since the observations are directly related to this context vector (since
    they are the coefficients of the quadratic) and makes the problem super easy
    to solve.
    """

    hparams: ClassVar[
        Type[QuadraticsTaskWithContextHparams]
    ] = QuadraticsTaskWithContextHparams

    def __init__(
        self,
        a2: float = None,
        a1: float = None,
        a0: float = None,
        task_id: int = 0,
        rng: np.random.Generator = None,
    ):
        super().__init__(a2=a2, a1=a1, a0=a0, task_id=task_id, rng=rng)
        # Fix those values so that all samples have these attributes set to
        # these values.
        self.fixed_values.update(a2=self.a2, a1=self.a1, a0=self.a0)

    @property
    def context_vector(self) -> np.ndarray:
        return np.asfarray([self.a2, self.a1, self.a0])

    @property
    def hash(self) -> str:
        # TODO: Return a unique "hash"/id/key for this task
        from warmstart.utils import compute_identity
        from dataclasses import is_dataclass, asdict

        if is_dataclass(self):
            return compute_identity(**asdict(self))
        # space = self.space
        return compute_identity(
            space=self.space,
            fixed_values=self.fixed_values,
            a0=self.a0,
            a1=self.a1,
            a2=self.a2,
            task_id=self.task_id,
        )
