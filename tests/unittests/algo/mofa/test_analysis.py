import itertools

import numpy as np
import pandas as pd
import pytest

from orion.algo.mofa.mofa import (
    MOFA,
    get_factorial_importance_analysis,
    get_factorial_performance_analysis,
    select_new_region_of_interest,
)
from orion.algo.mofa.transformer import Transformer
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.transformer import build_required_space


@pytest.fixture
def space():
    # Use a search space with dimensions of any type.
    original_space = SpaceBuilder().build(
        {
            "unr": "uniform(0, 10)",
            "uni": "uniform(0, 20, discrete=True)",
            "uns": "uniform(0, 5, shape=[2, 3])",
            "lur": "loguniform(1e-5, 0.1)",
            "lui": "loguniform(1, 100, discrete=True)",
            "cat": 'choices(["what", "ever", 0.33])',
        }
    )

    return build_required_space(
        original_space,
        type_requirement=MOFA.requires_type,
        shape_requirement=MOFA.requires_shape,
        dist_requirement=MOFA.requires_dist,
    )


def get_n_completed_trials(space, n):
    trials = space.sample(n)
    for i, trial in enumerate(trials):
        trial.results.append(trial.Result(name="objective", value=i, type="objective"))
        trial.status = "completed"

    return trials


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_generate_olh_perf_table(space, n_levels):
    N = 20
    transformer = Transformer(space, n_levels)
    trials = get_n_completed_trials(space, N)

    olh_perf_table = transformer.generate_olh_perf_table(trials)
    assert olh_perf_table.to_numpy()[:, :-1].min() == 0.0
    assert olh_perf_table.to_numpy()[:, :-1].max() == 1.0
    assert (olh_perf_table["objective"] == range(N)).all()


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_collapse_levels(space, n_levels):
    N = 20
    transformer = Transformer(space, n_levels)
    trials = get_n_completed_trials(space, N)

    olh_perf_table = transformer.generate_olh_perf_table(trials)
    oa_table = transformer._collapse_levels(olh_perf_table, n_levels)
    assert oa_table.to_numpy()[:, :-1].min() == 1
    assert oa_table.to_numpy()[:, :-1].max() == n_levels
    assert (oa_table["objective"] == range(N)).all()


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_generate_oa_table(space, n_levels):
    N = 20
    transformer = Transformer(space, n_levels)
    trials = get_n_completed_trials(space, N)
    oa_table = transformer.generate_oa_table(trials)
    assert oa_table.to_numpy()[:, :-1].min() == 1
    assert oa_table.to_numpy()[:, :-1].max() == n_levels
    assert (oa_table["objective"] == range(N)).all()


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_factorial_performance_analysis(space, n_levels):
    N = 20
    transformer = Transformer(space, n_levels)
    trials = get_n_completed_trials(space, N)
    oa_table = transformer.generate_oa_table(trials)

    factorial_performance_analysis = get_factorial_performance_analysis(
        oa_table, space, n_levels
    )

    assert factorial_performance_analysis.columns.to_list() == ["level"] + list(
        space.keys()
    )

    for key in space.keys():
        df = oa_table[[key, "objective"]].groupby([key]).mean()
        levels = df.index.values.tolist()
        ground_truth = df["objective"].to_list()
        for i in range(n_levels):
            if len(levels) - 1 < i or levels[i] > i + 1:
                levels.insert(i, i + 1)
                ground_truth.insert(i, float("inf"))

        assert factorial_performance_analysis[key].to_list() == ground_truth


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_factorial_importance_analysis(space, n_levels):
    N = 20
    transformer = Transformer(space, n_levels)
    trials = get_n_completed_trials(space, N)
    oa_table = transformer.generate_oa_table(trials)

    factorial_performance_analysis = get_factorial_performance_analysis(
        oa_table, space, n_levels
    )

    factorial_importance_analysis = get_factorial_importance_analysis(
        factorial_performance_analysis, space
    )

    variance = np.ma.masked_invalid(
        factorial_performance_analysis[space.keys()].to_numpy()
    ).var(0)
    assert variance.argmin() == factorial_importance_analysis["importance"].argmin()
    assert variance.argmax() == factorial_importance_analysis["importance"].argmax()


@pytest.mark.parametrize(
    "n_levels,threshold", itertools.product([2, 3, 5], [0.05, 0.1, 0.2])
)
def test_select_new_region_of_interest(space, n_levels, threshold):
    EPSILON = 1e-5
    factorial_importance_analysis = []
    for i, key in enumerate(space.keys()):
        factorial_importance_analysis.append(
            (key, i % n_levels + 1, 0.1 * (i % n_levels))
        )
    factorial_importance_analysis = pd.DataFrame(
        factorial_importance_analysis, columns=["param", "best_level", "importance"]
    )

    new_space, frozen_param_values = select_new_region_of_interest(
        factorial_importance_analysis, space, threshold, n_levels
    )

    tested_dim = set()
    for _, row in factorial_importance_analysis[
        factorial_importance_analysis["importance"] < threshold
    ].iterrows():
        tested_dim.add(str(row["param"]))
        assert row["param"] in frozen_param_values
        assert (
            frozen_param_values[row["param"]]
            == sum(space[row["param"]].interval()) / 2.0
        )

    for _, row in factorial_importance_analysis[
        factorial_importance_analysis["importance"] >= threshold
    ].iterrows():
        dim_name = str(row["param"])
        tested_dim.add(dim_name)
        assert dim_name in new_space
        assert new_space[dim_name].type == space[dim_name].type
        # New ROI in smaller than original space
        assert (
            space[dim_name].interval()[0] - EPSILON <= new_space[dim_name].interval()[0]
        )
        assert (
            space[dim_name].interval()[1] + EPSILON >= new_space[dim_name].interval()[1]
        )
        low, high = space[dim_name].interval()
        intervals = (high - low) / n_levels
        new_low = low + intervals * (int(row["best_level"]) - 1)
        # New ROI is correct level
        assert new_space[dim_name].interval()[0] == new_low
        assert new_space[dim_name].interval()[1] == new_low + intervals

    assert tested_dim == set(space.keys())
