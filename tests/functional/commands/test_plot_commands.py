#!/usr/bin/env python
"""Perform a functional test of the plot command."""
import random

import pytest

import orion.core.cli
import orion.plotting.backend_plotly
from orion.core.cli.plot import IMAGE_TYPES, VALID_TYPES
from orion.plotting.base import SINGLE_EXPERIMENT_PLOTS
from orion.storage.base import setup_storage
from orion.testing import AssertNewFile


def test_no_args(capsys):
    """Test that help is printed when no args are given."""
    with pytest.raises(SystemExit):
        orion.core.cli.main(["plot"])

    captured = capsys.readouterr().err

    assert "usage:" in captured
    assert "the following arguments are required: kind" in captured
    assert "Traceback" not in captured


def test_no_name(capsys):
    """Try to run the command without providing an experiment name"""
    returncode = orion.core.cli.main(["plot", "lpi"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured == "Error: No name provided for the experiment.\n"


def test_no_experiments(capsys, storage):
    """Test plot error message if experiment not found."""
    returncode = orion.core.cli.main(["plot", "regret", "--name", "idontexist"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured.startswith(
        "Error: No experiment with given name 'idontexist' and version '*'"
    )


def test_invalid_kind(capsys, single_with_trials):
    """Test that invalid kind leads to correct error message"""
    with pytest.raises(SystemExit):
        returncode = orion.core.cli.main(["plot", "blorg"])

    captured = capsys.readouterr().err

    assert "argument kind: invalid choice: 'blorg' (choose from 'lpi'," in captured


def test_default_type(single_with_trials):
    """Test that default type of image is used properly"""
    filename = "test_single_exp-v1_regret.png"
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            ["plot", "regret", "--name", "test_single_exp"]
        )
        assert returncode == 0


def test_invalid_type(capsys, single_with_trials):
    """Test that invalid type leads to correct error message"""
    with pytest.raises(SystemExit):
        returncode = orion.core.cli.main(
            ["plot", "regret", "--name", "test_single_exp", "--type", "boom"]
        )

    captured = capsys.readouterr().err

    assert "-t/--type: invalid choice: 'boom' (choose from 'png'" in captured


@pytest.mark.parametrize("out_type", VALID_TYPES)
def test_type(out_type, single_with_trials):
    """Test that given type of image is used properly"""

    filename = f"test_single_exp-v1_regret.{out_type}"
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            ["plot", "regret", "--name", "test_single_exp", "--type", out_type]
        )
        assert returncode == 0


def test_output(single_with_trials):
    """Test that output is used properly"""

    output = "custom"
    filename = f"{output}.png"
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            ["plot", "regret", "--name", "test_single_exp", "--output", output]
        )
        assert returncode == 0


def test_output_type(single_with_trials):
    """Test that output is used properly with given type"""

    for output in ["custom", "custom.with.points", "custombadjpg"]:
        filename = f"{output}.jpg"
        with AssertNewFile(filename):
            returncode = orion.core.cli.main(
                [
                    "plot",
                    "regret",
                    "--name",
                    "test_single_exp",
                    "--output",
                    output,
                    "--type",
                    "jpg",
                ]
            )
            assert returncode == 0


def test_output_override_type(single_with_trials):
    """Test that output with type will override --type"""

    output = "custom.jpg"
    filename = output
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            [
                "plot",
                "regret",
                "--name",
                "test_single_exp",
                "--output",
                output,
                "--type",
                "pdf",
            ]
        )
        assert returncode == 0


@pytest.mark.parametrize("kind", SINGLE_EXPERIMENT_PLOTS.keys())
def test_output_default_kind(monkeypatch, single_with_trials, kind):
    """Test that output default is used properly and kind is called"""

    plotting_function = getattr(orion.plotting.backend_plotly, kind)

    def mock_plot(*args, **kwargs):
        mock_plot.called = True
        return plotting_function(*args, **kwargs)

    mock_plot.called = False

    monkeypatch.setattr(f"orion.plotting.backend_plotly.{kind}", mock_plot)

    assert len(setup_storage().fetch_trials(uid=single_with_trials["_id"])) > 0

    filename = f"test_single_exp-v1_{kind}.png"
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            [
                "plot",
                kind,
                "--name",
                "test_single_exp",
            ]
        )
        assert returncode == 0

    assert mock_plot.called


def test_scale(monkeypatch, single_with_trials):
    """Test that scale is passed properly"""

    SCALE = random.uniform(0, 3)

    def check_args(self, output, scale, **kwargs):
        assert scale == SCALE

    monkeypatch.setattr("plotly.graph_objs._figure.Figure.write_image", check_args)

    returncode = orion.core.cli.main(
        ["plot", "regret", "--name", "test_single_exp", "--scale", str(SCALE)]
    )
    assert returncode == 0


@pytest.mark.parametrize("kind", SINGLE_EXPERIMENT_PLOTS.keys())
def test_no_trials(one_experiment, kind):
    """Test plotting works with empty experiments"""

    assert setup_storage().fetch_trials(uid=one_experiment["_id"]) == []

    filename = f"test_single_exp-v1_{kind}.png"
    with AssertNewFile(filename):
        returncode = orion.core.cli.main(
            [
                "plot",
                kind,
                "--name",
                "test_single_exp",
            ]
        )
        assert returncode == 0


def test_html(monkeypatch, single_with_trials):
    """Test html is saved properly"""

    def mock_html(*args, **kwargs):
        mock_html.called = True
        return ""

    mock_html.called = False

    monkeypatch.setattr("plotly.graph_objs._figure.Figure.write_html", mock_html)

    returncode = orion.core.cli.main(
        ["plot", "regret", "--name", "test_single_exp", "--type", "html"]
    )
    assert returncode == 0

    assert mock_html.called


def test_json(monkeypatch, single_with_trials):
    """Test json is saved properly"""

    def mock_json(*args, **kwargs):
        mock_json.called = True
        return ""

    mock_json.called = False

    monkeypatch.setattr("plotly.graph_objs._figure.Figure.to_json", mock_json)

    returncode = orion.core.cli.main(
        ["plot", "regret", "--name", "test_single_exp", "--type", "json"]
    )
    assert returncode == 0

    assert mock_json.called


@pytest.mark.parametrize("out_type", IMAGE_TYPES)
def test_write_image(monkeypatch, single_with_trials, out_type):
    """Test all types for images use write_image"""

    def mock_write_image(*args, **kwargs):
        mock_write_image.called = True
        return ""

    mock_write_image.called = False

    monkeypatch.setattr(
        "plotly.graph_objs._figure.Figure.write_image", mock_write_image
    )

    returncode = orion.core.cli.main(
        ["plot", "regret", "--name", "test_single_exp", "--type", out_type]
    )
    assert returncode == 0

    assert mock_write_image.called
