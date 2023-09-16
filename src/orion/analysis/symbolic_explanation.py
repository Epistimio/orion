"""
Module for symbolic explanation.

Large chunk of codes are copied from this repository: https://github.com/automl/symbolic-explanations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import sympy
from gplearn import functions
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

from orion.algo.space import Categorical, Space
from orion.analysis.base import flatten_numpy, flatten_params, to_numpy, train_regressor
from orion.core.utils import format_trials
from orion.core.worker.transformer import build_required_space

# TODO: Add dependencies:
#       gplearn
#       sympy

logger = logging.getLogger(__name__)


# TODO: May need to set a_max as a parameter so user can change it.
def safe_exp(x):
    return np.clip(x, a_min=None, a_max=100000)


@dataclass
class SymbolicRegressorParams:
    population_size: int = 5000
    generations: int = 20
    metric: str = "rmse"
    parsimony_coefficient: float = 0.0001
    random_state: int | None = None
    verbose: int = 1
    function_set: list = field(
        default_factory=lambda: [
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "sin",
            "cos",
            "abs",
            make_function(function=safe_exp, arity=1, name="exp"),
        ]
    )


SymbolicRegressorParams(metric="test")


def convert_symb(symb, space: Space, n_decimals: int = None) -> sympy.core.expr:
    """
    Convert a fitted symbolic regression to a simplified and potentially rounded mathematical expression.
    Warning: eval is used in this function, thus it should not be used on unsanitized input (see
    https://docs.sympy.org/latest/modules/core.html?highlight=eval#module-sympy.core.sympify).

    Parameters
    ----------
    symb: Fitted symbolic regressor to find a simplified expression for.
    n_dim: Number of input dimensions. If input has only a single dimension, X0 in expression is exchanged by x.
    n_decimals: If set, round floats in the expression to this number of decimals.

    Returns
    -------
    symb_conv: Converted mathematical expression.
    """

    # sqrt is protected function in gplearn, always returning sqrt(abs(x))
    sqrt_pos = []
    prev_sqrt_inserts = 0
    for i, f in enumerate(symb._program.program):
        if isinstance(f, functions._Function) and f.name == "sqrt":
            sqrt_pos.append(i)
    for i in sqrt_pos:
        symb._program.program.insert(i + prev_sqrt_inserts + 1, functions.abs1)
        prev_sqrt_inserts += 1

    # log is protected function in gplearn, always returning sqrt(abs(x))
    log_pos = []
    prev_log_inserts = 0
    for i, f in enumerate(symb._program.program):
        if isinstance(f, functions._Function) and f.name == "log":
            log_pos.append(i)
    for i in log_pos:
        symb._program.program.insert(i + prev_log_inserts + 1, functions.abs1)
        prev_log_inserts += 1

    symb_str = str(symb._program)

    converter = {
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "mul": lambda x, y: x * y,
        "add": lambda x, y: x + y,
        "neg": lambda x: -x,
        "pow": lambda x, y: x**y,
    }

    converter.update(
        {
            f"X{i}": sympy.symbols(dim.name, real=True)
            for i, dim in enumerate(space.values())
        }
    )

    if symb._program.length_ > 300:
        print(
            f"Expression of length {symb._program._length} too long to convert, return raw string."
        )
        return symb_str

    symb_conv = sympy.sympify(
        symb_str.replace("[", "").replace("]", ""), locals=converter
    )
    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_conv):
            if isinstance(a, sympy.core.numbers.Float):
                symb_conv = symb_conv.subs(a, round(a, n_decimals))

    return symb_conv


def simplify_formula(
    symb_model: SymbolicRegressor, space: Space, n_decimals: int = 3
) -> str:
    conv_expr = convert_symb(symb_model, space=space, n_decimals=n_decimals)

    return conv_expr


def symbolic_explanation(
    trials: pd.DataFrame,
    space: Space,
    params: List[str] | None = None,
    model="RandomForestRegressor",
    n_samples: int = 50,
    sampling_seed: int | None = None,
    timeout: float = 300,
    symbolic_regressor_params: SymbolicRegressorParams = SymbolicRegressorParams(),
    **kwargs,
) -> SymbolicRegressor:
    """
    Calculates a symbolic explanation of the effect of parameters
    on the objective based on a collection of
    :class:`orion.core.worker.trial.Trial`.

    For more information on the method,
    see original paper at https://openreview.net/forum?id=JQwAc91sg_x.

    Segel, Sarah, et al. "Symbolic explanations for hyperparameter
    optimization." AutoML Conference 2023.

    Parameters
    ----------
    trials: DataFrame or dict
        A dataframe of trials containing, at least, the columns 'objective' and 'id'. Or a dict
        equivalent.

    space: Space object
        A space object from an experiment.

    params: list of str, optional
        The parameters to include in the computation. All parameters are included by default.

    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)

    n_samples: int
        Number of samples to randomly generate for fitting the surrogate model.
        Default is 50.

    sampling_seed: int
        Seed used to sample the points for fitting the surrogate model.

    timeout: float
        Number of seconds before the evolutionary algorithm is stopped.

    symbolic_regressor_params: SymbolicRegressorParams
        Dataclass for the parameters for the ``SymbolicRegressor``.

    **kwargs
        Arguments for the regressor model.

    Returns
    -------
    SymbolicRegressor
        A SymbolicRegressor fitted on the trials.
    """

    # TODO: Need to handle multi-fidelity. Maybe only pick the highest fidelity...

    # TODO: Validate that the history (`trials`) is large enough to make sense compared to the number of sampled
    # points from the surrogate.

    if any(isinstance(dim, Categorical) for dim in space):
        raise ValueError(
            "Symbolic explanation does not support categorical dimensions yet."
        )

    params = flatten_params(space, params)

    flattened_space = build_required_space(
        space,
        dist_requirement="linear",
        type_requirement="numerical",
        shape_requirement="flattened",
    )

    if trials.empty or trials.shape[0] == 0:
        return {}

    data = to_numpy(trials, space)
    data = flatten_numpy(data, flattened_space)
    model = train_regressor(model, data, **kwargs)

    # Sample random points for fitting the symbolic regressor.
    data = [
        format_trials.trial_to_tuple(trial, flattened_space)
        for trial in flattened_space.sample(n_samples, seed=sampling_seed)
    ]
    X_train = pd.DataFrame(data, columns=flattened_space.keys()).to_numpy()

    Y_train = model.predict(X_train)

    symbolic_regressor = SymbolicRegressor(
        population_size=symbolic_regressor_params.population_size,
        generations=symbolic_regressor_params.generations,
        function_set=symbolic_regressor_params.function_set,
        metric=symbolic_regressor_params.metric,
        parsimony_coefficient=symbolic_regressor_params.parsimony_coefficient,
        verbose=symbolic_regressor_params.verbose,
        random_state=symbolic_regressor_params.random_state,
    )

    symbolic_regressor.fit(X_train, Y_train)

    return symbolic_regressor
