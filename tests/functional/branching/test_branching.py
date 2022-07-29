#!/usr/bin/env python
"""Perform a functional test for branching."""

import logging
import os

import pytest
import yaml

import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
from orion.storage.base import setup_storage


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


@pytest.fixture
def init_full_x(orionstate, monkeypatch):
    """Init original experiment"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    name = "full_x"
    orion.core.cli.main(
        (
            "hunt --init-only -n {name} --config orion_config.yaml ./black_box.py "
            "-x~uniform(-10,10)"
        )
        .format(name=name)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {name} script -x=0".split(" "))


@pytest.fixture
def init_no_evc(monkeypatch):
    """Add y dimension but overwrite original"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    name = "full_x"
    branch = "wont_exist"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "./black_box_with_y.py "
            "-x~uniform(-10,10) "
            "-y~+uniform(-10,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {name} script -x=1 -y=1".split(" "))
    orion.core.cli.main(f"insert -n {name} script -x=-1 -y=1".split(" "))
    orion.core.cli.main(f"insert -n {name} script -x=1 -y=-1".split(" "))
    orion.core.cli.main(f"insert -n {name} script -x=-1 -y=-1".split(" "))


@pytest.fixture
def init_full_x_full_y(init_full_x):
    """Add y dimension to original"""
    print("init_full_x_full_y start")

    name = "full_x"
    branch = "full_x_full_y"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box_with_y.py "
            "-x~uniform(-10,10) "
            "-y~+uniform(-10,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=1 -y=1".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-1 -y=1".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=1 -y=-1".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-1 -y=-1".split(" "))


@pytest.fixture
def init_half_x_full_y(init_full_x_full_y):
    """Change x's prior to full x and full y experiment"""
    name = "full_x_full_y"
    branch = "half_x_full_y"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} "
            "--enable-evc "
            "./black_box_with_y.py "
            "-x~+uniform(0,10) "
            "-y~uniform(-10,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=2 -y=2".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=2 -y=-2".split(" "))


@pytest.fixture
def init_full_x_half_y(init_full_x_full_y):
    """Change y's prior to full x and full y experiment"""
    name = "full_x_full_y"
    branch = "full_x_half_y"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} "
            "--enable-evc "
            "./black_box_with_y.py "
            "-x~uniform(-10,10) "
            "-y~+uniform(0,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=3 -y=3".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-3 -y=3".split(" "))


@pytest.fixture
def init_full_x_rename_y_z(init_full_x_full_y):
    """Rename y from full x full y to z"""
    name = "full_x_full_y"
    branch = "full_x_rename_y_z"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box_with_z.py -x~uniform(-10,10) -y~>z -z~uniform(-10,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=4 -z=4".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-4 -z=4".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=4 -z=-4".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-4 -z=-4".split(" "))


@pytest.fixture
def init_full_x_rename_half_y_half_z(init_full_x_half_y):
    """Rename y from full x half y to z"""
    name = "full_x_half_y"
    branch = "full_x_rename_half_y_half_z"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box_with_z.py -x~uniform(-10,10) -y~>z -z~uniform(0,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=5 -z=5".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-5 -z=5".split(" "))


@pytest.fixture
def init_full_x_rename_half_y_full_z(init_full_x_half_y):
    """Rename y from full x half y to full z (rename + changed prior)"""
    name = "full_x_half_y"
    branch = "full_x_rename_half_y_full_z"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box_with_z.py "
            "-x~uniform(-10,10) -y~>z "
            "-z~+uniform(-10,10,default_value=1)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=6 -z=6".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-6 -z=6".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=6 -z=-6".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-6 -z=-6".split(" "))


@pytest.fixture
def init_full_x_remove_y(init_full_x_full_y):
    """Remove y from full x full y"""
    name = "full_x_full_y"
    branch = "full_x_remove_y"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box.py "
            "-x~uniform(-10,10) -y~-"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=7".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-7".split(" "))


@pytest.fixture
def init_full_x_full_y_add_z_remove_y(init_full_x_full_y):
    """Remove y from full x full y and add z"""
    name = "full_x_full_y"
    branch = "full_x_full_z_remove_y"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10) "
            "-z~uniform(-20,10,default_value=0)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=7 -z=2".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-7 -z=2".split(" "))


@pytest.fixture
def init_full_x_remove_z(init_full_x_rename_y_z):
    """Remove z from full x full z"""
    name = "full_x_rename_y_z"
    branch = "full_x_remove_z"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box.py "
            "-x~uniform(-10,10) -z~-"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=8".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-8".split(" "))


@pytest.fixture
def init_full_x_remove_z_default_4(init_full_x_rename_y_z):
    """Remove z from full x full z and give a default value of 4"""
    name = "full_x_rename_y_z"
    branch = "full_x_remove_z_default_4"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box.py "
            "-x~uniform(-10,10) -z~-4"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=9".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-9".split(" "))


@pytest.fixture
def init_full_x_new_algo(init_full_x):
    """Remove z from full x full z and give a default value of 4"""
    name = "full_x"
    branch = "full_x_new_algo"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} "
            "--algorithm-change --config new_algo_config.yaml "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=1.1".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-1.1".split(" "))


@pytest.fixture
def init_full_x_new_cli(init_full_x):
    """Change commandline call"""
    name = "full_x"
    branch = "full_x_new_cli"
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --cli-change-type noeffect "
            "--enable-evc "
            "./black_box_new.py -x~uniform(-10,10) --a-new argument"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {branch} script -x=1.2".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-1.2".split(" "))


@pytest.fixture
def init_full_x_ignore_cli(init_full_x):
    """Use the --non-monitored-arguments argument"""
    name = "full_x_with_new_opt"
    orion.core.cli.main(
        (
            "hunt --init-only -n {name} --config orion_config.yaml "
            "--enable-evc "
            "./black_box_new.py "
            "-x~uniform(-10,10)"
        )
        .format(name=name)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {name} script -x=0".split(" "))

    orion.core.cli.main(
        (
            "hunt --init-only -n {name} --non-monitored-arguments a-new "
            "--config orion_config.yaml "
            "--enable-evc "
            "./black_box_new.py "
            "-x~uniform(-10,10) --a-new argument"
        )
        .format(name=name)
        .split(" ")
    )
    orion.core.cli.main(f"insert -n {name} script -x=1.2".split(" "))
    orion.core.cli.main(f"insert -n {name} script -x=-1.2".split(" "))


@pytest.fixture
def init_full_x_new_config(init_full_x, tmp_path, caplog):
    """Add configuration script"""
    name = "full_x"
    branch = "full_x_new_config"

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump(
            {"new_arg": "some-value", "y": "orion~uniform(-10, 10, default_value=0)"}
        )
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        orion.core.cli.main(
            (
                "hunt --enable-evc --init-only -n {branch} --branch-from {name} "
                "--cli-change-type noeffect "
                "--config-change-type unsure "
                "--user-script-config custom-config "
                "./black_box_new.py -x~uniform(-10,10) --custom-config {config_file}"
            )
            .format(name=name, branch=branch, config_file=config_file)
            .split(" ")
        )
        # For parent experiment
        assert "User script config: config" in caplog.text
        # For child experiment
        assert "User script config: custom-config" in caplog.text

    orion.core.cli.main(f"insert -n {branch} script -x=1.2 -y=2".split(" "))
    orion.core.cli.main(f"insert -n {branch} script -x=-1.2 -y=3".split(" "))


@pytest.fixture
def init_entire(
    init_half_x_full_y,  # 1.1.1
    init_full_x_rename_half_y_half_z,  # 1.1.2.1
    init_full_x_rename_half_y_full_z,  # 1.1.2.2
    init_full_x_remove_y,  # 1.1.4
    init_full_x_remove_z,  # 1.1.3.1
    init_full_x_remove_z_default_4,
):  # 1.1.3.2
    """Initialize all experiments"""


def get_name_value_pairs(trials):
    """Turn parameters into pairs for easy comparisons"""
    pairs = []
    for trial in trials:
        pairs.append([])
        for param in trial._params:
            pairs[-1].append((param.name, param.value))

        pairs[-1] = tuple(pairs[-1])

    return tuple(pairs)


def test_init(init_full_x):
    """Test if original experiment contains trial 0"""
    experiment = experiment_builder.load(name="full_x")

    assert experiment.refers["adapter"].configuration == []
    assert experiment.space.configuration == {"/x": "uniform(-10, 10)"}

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 0),),)


def test_no_evc_overwrite(orionstate, init_no_evc):
    """Test that the experiment config is overwritten if --enable-evc is not passed"""
    storage = setup_storage()
    assert len(storage.fetch_experiments({})) == 1
    experiment = experiment_builder.load(name="full_x")

    assert experiment.refers["adapter"].configuration == []
    assert experiment.space.configuration == {
        "/x": "uniform(-10, 10)",
        "/y": "uniform(-10, 10, default_value=1)",
    }

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == (
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", -1), ("/y", -1)),
    )


def test_full_x_full_y(init_full_x_full_y):
    """Test if full x full y is properly initialized and can fetch original trial"""
    experiment = experiment_builder.load(name="full_x_full_y")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == (
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", -1), ("/y", -1)),
    )

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/y", 1)),
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", -1), ("/y", -1)),
    )


def test_half_x_full_y(init_half_x_full_y):
    """Test if half x full y is properly initialized and can fetch from its 2 parents"""
    experiment = experiment_builder.load(name="half_x_full_y")

    assert experiment.refers["adapter"].configuration == [
        {
            "of_type": "dimensionpriorchange",
            "name": "/x",
            "old_prior": "uniform(-10, 10)",
            "new_prior": "uniform(0, 10)",
        }
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 2), ("/y", 2)), (("/x", 2), ("/y", -2)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/y", 1)),
        (("/x", 1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", 2), ("/y", 2)),
        (("/x", 2), ("/y", -2)),
    )


def test_full_x_half_y(init_full_x_half_y):
    """Test if full x half y is properly initialized and can fetch from its 2 parents"""
    experiment = experiment_builder.load(name="full_x_half_y")

    assert experiment.refers["adapter"].configuration == [
        {
            "of_type": "dimensionpriorchange",
            "name": "/y",
            "old_prior": "uniform(-10, 10, default_value=1)",
            "new_prior": "uniform(0, 10, default_value=1)",
        }
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 3), ("/y", 3)), (("/x", -3), ("/y", 3)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/y", 1)),
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 3), ("/y", 3)),
        (("/x", -3), ("/y", 3)),
    )


def test_full_x_rename_y_z(init_full_x_rename_y_z):
    """Test if full x full z is properly initialized and can fetch from its 2 parents"""
    experiment = experiment_builder.load(name="full_x_rename_y_z")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "commandlinechange", "change_type": "noeffect"},
        {"of_type": "dimensionrenaming", "old_name": "/y", "new_name": "/z"},
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == (
        (("/x", 4), ("/z", 4)),
        (("/x", -4), ("/z", 4)),
        (("/x", 4), ("/z", -4)),
        (("/x", -4), ("/z", -4)),
    )

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/z", 1)),
        (("/x", 1), ("/z", 1)),
        (("/x", -1), ("/z", 1)),
        (("/x", 1), ("/z", -1)),
        (("/x", -1), ("/z", -1)),
        (("/x", 4), ("/z", 4)),
        (("/x", -4), ("/z", 4)),
        (("/x", 4), ("/z", -4)),
        (("/x", -4), ("/z", -4)),
    )


def test_full_x_rename_half_y_half_z(init_full_x_rename_half_y_half_z):
    """Test if full x half z is properly initialized and can fetch from its 3 parents"""
    experiment = experiment_builder.load(name="full_x_rename_half_y_half_z")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "commandlinechange", "change_type": "noeffect"},
        {"of_type": "dimensionrenaming", "old_name": "/y", "new_name": "/z"},
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 5), ("/z", 5)), (("/x", -5), ("/z", 5)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/z", 1)),
        (("/x", 1), ("/z", 1)),
        (("/x", -1), ("/z", 1)),
        (("/x", 3), ("/z", 3)),
        (("/x", -3), ("/z", 3)),
        (("/x", 5), ("/z", 5)),
        (("/x", -5), ("/z", 5)),
    )


def test_full_x_rename_half_y_full_z(init_full_x_rename_half_y_full_z):
    """Test if full x half->full z is properly initialized and can fetch from its 3 parents"""
    experiment = experiment_builder.load(name="full_x_rename_half_y_full_z")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "commandlinechange", "change_type": "noeffect"},
        {"of_type": "dimensionrenaming", "old_name": "/y", "new_name": "/z"},
        {
            "of_type": "dimensionpriorchange",
            "name": "/z",
            "old_prior": "uniform(0, 10, default_value=1)",
            "new_prior": "uniform(-10, 10, default_value=1)",
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == (
        (("/x", 6), ("/z", 6)),
        (("/x", -6), ("/z", 6)),
        (("/x", 6), ("/z", -6)),
        (("/x", -6), ("/z", -6)),
    )

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/z", 1)),
        (("/x", 1), ("/z", 1)),
        (("/x", -1), ("/z", 1)),
        (("/x", 3), ("/z", 3)),
        (("/x", -3), ("/z", 3)),
        (("/x", 6), ("/z", 6)),
        (("/x", -6), ("/z", 6)),
        (("/x", 6), ("/z", -6)),
        (("/x", -6), ("/z", -6)),
    )


def test_full_x_remove_y(init_full_x_remove_y):
    """Test if full x removed y is properly initialized and can fetch from its 2 parents"""
    experiment = experiment_builder.load(name="full_x_remove_y")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "commandlinechange", "change_type": "noeffect"},
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 7),), (("/x", -7),))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0),),
        (("/x", 1),),
        (("/x", -1),),
        (("/x", 7),),
        (("/x", -7),),
    )


def test_full_x_full_y_add_z_remove_y(init_full_x_full_y_add_z_remove_y):
    """Test that if z is added and y removed at the same time, both are correctly detected"""
    experiment = experiment_builder.load(name="full_x_full_z_remove_y")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "commandlinechange", "change_type": "noeffect"},
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/z", "type": "real", "value": 0},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 7), ("/z", 2)), (("/x", -7), ("/z", 2)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == (
        (("/x", 0), ("/z", 0)),
        (("/x", 1), ("/z", 0)),
        (("/x", -1), ("/z", 0)),
        (("/x", 7), ("/z", 2)),
        (("/x", -7), ("/z", 2)),
    )


def test_full_x_remove_z(init_full_x_remove_z):
    """Test if full x removed z is properly initialized and can fetch from 2 of its 3 parents"""
    experiment = experiment_builder.load(name="full_x_remove_z")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/z", "type": "real", "value": 1},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 8),), (("/x", -8),))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    # Note that full_x_rename_y_z are filtered out because default_value=1
    assert pairs == (
        (("/x", 0),),
        (("/x", 1),),
        (("/x", -1),),
        (("/x", 8),),
        (("/x", -8),),
    )


def test_full_x_remove_z_default_4(init_full_x_remove_z_default_4):
    """Test if full x removed z  (default 4) is properly initialized and can fetch
    from 1 of its 3 parents
    """
    experiment = experiment_builder.load(name="full_x_remove_z_default_4")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/z", "type": "real", "value": 4.0},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((("/x", 9),), (("/x", -9),))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    # Note that full_x and full_x_full_y are filtered out because default_value=4
    assert pairs == ((("/x", 4),), (("/x", -4),), (("/x", 9),), (("/x", -9),))


def test_entire_full_x_full_y(init_entire):
    """Test if full x full y can fetch from its parent and all children"""
    experiment = experiment_builder.load(name="full_x_full_y")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
    ]

    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == (
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", -1), ("/y", -1)),
    )

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert set(pairs) == {
        (("/x", 0), ("/y", 1)),
        # full_x_full_y
        (("/x", 1), ("/y", 1)),
        (("/x", -1), ("/y", 1)),
        (("/x", 1), ("/y", -1)),
        (("/x", -1), ("/y", -1)),
        # half_x_full_y
        (("/x", 2), ("/y", 2)),
        (("/x", 2), ("/y", -2)),
        # full_x_half_y
        (("/x", 3), ("/y", 3)),
        (("/x", -3), ("/y", 3)),
        # full_x_rename_y_z
        (("/x", 4), ("/y", 4)),
        (("/x", -4), ("/y", 4)),
        (("/x", 4), ("/y", -4)),
        (("/x", -4), ("/y", -4)),
        # full_x_rename_half_y_half_z
        (("/x", 5), ("/y", 5)),
        (("/x", -5), ("/y", 5)),
        # full_x_rename_half_y_full_z
        (("/x", 6), ("/y", 6)),
        (("/x", -6), ("/y", 6)),
        # full_x_remove_y
        (("/x", 7), ("/y", 1)),
        (("/x", -7), ("/y", 1)),
        # full_x_remove_z
        (("/x", 8), ("/y", 1)),
        (("/x", -8), ("/y", 1)),
        # full_x_remove_z_default_4
        (("/x", 9), ("/y", 4)),
        (("/x", -9), ("/y", 4)),
    }


def test_run_entire_full_x_full_y(init_entire):
    """Test if branched experiment can be executed without triggering a branching event again"""
    experiment = experiment_builder.load(name="full_x_full_y")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
    ]

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 23
    assert len(experiment.fetch_trials()) == 4

    code = orion.core.cli.main(
        (
            "-vv hunt --max-trials 30 --pool-size 1 -n full_x_full_y "
            "./black_box_with_y.py "
            "-x~uniform(-10,10) "
            "-y~uniform(-10,10,default_value=1)"
        ).split(" ")
    )
    assert code == 0

    # Current experiments now contains all trials because
    # trials from parent exps have been duplicated in current exp to reserve them
    # (See experiment.duplicate_pending_trials)
    assert len(experiment.fetch_trials(with_evc_tree=True)) == 30
    assert len(experiment.fetch_trials(with_evc_tree=False)) == 30


def test_run_entire_full_x_full_y_no_args(init_entire):
    """Test if branched experiment can be executed without script arguments"""
    experiment = experiment_builder.load(name="full_x_full_y")

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 23
    assert len(experiment.fetch_trials()) == 4

    orion.core.cli.main(
        ("-vv hunt --max-trials 30 --pool-size 1 -n full_x_full_y").split(" ")
    )

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 30
    assert len(experiment.fetch_trials(with_evc_tree=False)) == 30


def test_new_algo(init_full_x_new_algo):
    """Test that new algo conflict is automatically resolved"""
    experiment = experiment_builder.load(name="full_x_new_algo")

    assert experiment.refers["adapter"].configuration == [
        {"of_type": "algorithmchange"}
    ]

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 3
    assert len(experiment.fetch_trials()) == 2

    orion.core.cli.main(
        ("-vv hunt --max-trials 20 --pool-size 1 -n full_x_new_algo").split(" ")
    )

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 20
    assert len(experiment.fetch_trials(with_evc_tree=False)) == 20


def test_new_algo_not_resolved(init_full_x, capsys):
    """Test that new algo conflict is not automatically resolved"""
    name = "full_x"
    branch = "full_x_new_algo"
    error_code = orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} --config new_algo_config.yaml "
            "--manual-resolution "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )
    assert error_code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration is different and generates a branching event" in captured.err
    assert "gradient_descent" in captured.err


def test_ignore_cli(init_full_x_ignore_cli):
    """Test that a non-monitored parameter conflict is not generating a child"""
    name = "full_x"
    orion.core.cli.main(
        (
            "hunt --init-only -n {name} --non-monitored-arguments a-new "
            "--manual-resolution "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10)"
        )
        .format(name=name)
        .split(" ")
    )


@pytest.mark.usefixtures("init_full_x", "mock_infer_versioning_metadata")
def test_new_code_triggers_code_conflict(capsys):
    """Test that a different git hash is generating a child"""
    name = "full_x"
    error_code = orion.core.cli.main(
        (
            "hunt --init-only -n {name} "
            "--manual-resolution "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10)"
        )
        .format(name=name)
        .split(" ")
    )
    assert error_code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration is different and generates a branching event" in captured.err
    assert "--code-change-type" in captured.err


@pytest.mark.usefixtures("init_full_x", "mock_infer_versioning_metadata")
def test_new_code_triggers_code_conflict_with_name_only(capsys):
    """Test that a different git hash is generating a child, even if cmdline is not passed"""
    name = "full_x"
    error_code = orion.core.cli.main(
        ("hunt --init-only -n {name} --manual-resolution --enable-evc")
        .format(name=name)
        .split(" ")
    )
    assert error_code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration is different and generates a branching event" in captured.err
    assert "--code-change-type" in captured.err


@pytest.mark.usefixtures("init_full_x", "mock_infer_versioning_metadata")
def test_new_code_ignores_code_conflict():
    """Test that a different git hash is *not* generating a child if --ignore-code-changes"""
    name = "full_x"
    # Let it run for 2 trials to test consumer._validate_code_version too.
    error_code = orion.core.cli.main(
        (
            "hunt --worker-max-trials 2 -n {name} --ignore-code-changes "
            "--manual-resolution "
            "--enable-evc "
            "./black_box.py -x~uniform(-10,10)"
        )
        .format(name=name)
        .split(" ")
    )
    assert error_code == 0


@pytest.mark.usefixtures("init_full_x", "version_XYZ")
def test_new_orion_version_triggers_conflict(capsys):
    """Test that a different git hash is generating a child"""
    name = "full_x"
    error_code = orion.core.cli.main(
        ("hunt --init-only -n {name} --manual-resolution --enable-evc")
        .format(name=name)
        .split(" ")
    )
    assert error_code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration is different and generates a branching event" in captured.err
    assert "XYZ" in captured.err


def test_new_cli(init_full_x_new_cli):
    """Test that new cli conflict is automatically resolved"""
    experiment = experiment_builder.load(name="full_x_new_cli")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"}
    ]

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 3
    assert len(experiment.fetch_trials()) == 2

    orion.core.cli.main(
        ("-vv hunt --max-trials 20 --pool-size 1 -n full_x_new_cli").split(" ")
    )

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 20
    assert len(experiment.fetch_trials(with_evc_tree=False)) == 20


@pytest.mark.usefixtures("init_full_x")
def test_no_cli_no_branching():
    """Test that no branching occurs when using same code and not passing cmdline"""
    name = "full_x"
    error_code = orion.core.cli.main(
        ("hunt --init-only -n {name} --manual-resolution --enable-evc")
        .format(name=name)
        .split(" ")
    )
    assert error_code == 0


def test_new_script(init_full_x, monkeypatch):
    """Test that experiment can branch with new script path even if previous is not present"""

    name = "full_x"
    experiment = experiment_builder.load(name=name)

    # Mess with DB to change script path
    metadata = experiment.metadata
    metadata["user_script"] = "oh_oh_idontexist.py"
    metadata["user_args"][0] = "oh_oh_idontexist.py"
    metadata["parser"]["parser"]["arguments"][0][1] = "oh_oh_idontexist.py"
    setup_storage().update_experiment(experiment, metadata=metadata)

    orion.core.cli.main(
        (
            "hunt --enable-evc --init-only -n {name} --config orion_config.yaml ./black_box.py "
            "-x~uniform(-10,10) --some-new args"
        )
        .format(name=name)
        .split(" ")
    )

    new_experiment = experiment_builder.load(name=name)
    assert new_experiment.version == experiment.version + 1

    assert new_experiment.refers["adapter"].configuration == [
        {"change_type": "break", "of_type": "commandlinechange"}
    ]


def test_new_config(init_full_x_new_config, monkeypatch):
    """Test experiment branching with new config"""
    experiment = experiment_builder.load(name="full_x_new_config")

    assert experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/y", "type": "real", "value": 0},
        },
        {"change_type": "unsure", "of_type": "scriptconfigchange"},
    ]

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 3
    assert len(experiment.fetch_trials()) == 2


def test_missing_config(init_full_x_new_config, monkeypatch):
    """Test that experiment can branch with new config if previous is not present"""
    name = "full_x_new_config"
    experiment = experiment_builder.load(name=name)

    # Mess with DB to change config path
    metadata = experiment.metadata
    bad_config_file = "ho_ho_idontexist.yaml"
    config_file = metadata["parser"]["file_config_path"]
    metadata["parser"]["file_config_path"] = bad_config_file
    metadata["parser"]["parser"]["arguments"][2][1] = bad_config_file
    metadata["user_args"][3] = bad_config_file
    setup_storage().update_experiment(experiment, metadata=metadata)

    orion.core.cli.main(
        (
            "hunt --enable-evc --init-only -n {name} "
            "--cli-change-type noeffect "
            "--config-change-type unsure "
            "./black_box_new.py -x~uniform(-10,10) --config {config_file}"
        )
        .format(name=name, config_file=config_file)
        .split(" ")
    )

    new_experiment = experiment_builder.load(name=name)
    assert new_experiment.version == experiment.version + 1

    assert new_experiment.refers["adapter"].configuration == [
        {"change_type": "noeffect", "of_type": "commandlinechange"}
    ]


def test_missing_and_new_config(init_full_x_new_config, monkeypatch):
    """Test that experiment can branch with new config if previous is not present, with correct
    diff.
    """
    name = "full_x_new_config"
    experiment = experiment_builder.load(name=name)

    # Mess with DB to change config path
    metadata = experiment.metadata
    bad_config_file = "ho_ho_idontexist.yaml"
    config_file = metadata["parser"]["file_config_path"]
    metadata["parser"]["file_config_path"] = bad_config_file
    metadata["parser"]["parser"]["arguments"][2][1] = bad_config_file
    metadata["user_args"][3] = bad_config_file

    with open(config_file, "w") as f:
        f.write(
            yaml.dump(
                {
                    "new_arg": "some-new-value",
                    "y": "orion~uniform(-10, 20, default_value=0)",
                }
            )
        )

    setup_storage().update_experiment(experiment, metadata=metadata)

    orion.core.cli.main(
        (
            "hunt --enable-evc --init-only -n {name} "
            "--cli-change-type noeffect "
            "--config-change-type unsure "
            "./black_box_new.py -x~uniform(-10,10) --config {config_file}"
        )
        .format(name=name, config_file=config_file)
        .split(" ")
    )

    new_experiment = experiment_builder.load(name=name)
    assert new_experiment.version == experiment.version + 1

    assert new_experiment.refers["adapter"].configuration == [
        {
            "name": "/y",
            "new_prior": "uniform(-10, 20, default_value=0)",
            "of_type": "dimensionpriorchange",
            "old_prior": "uniform(-10, 10, default_value=0)",
        },
        {"change_type": "noeffect", "of_type": "commandlinechange"},
        {"change_type": "unsure", "of_type": "scriptconfigchange"},
    ]


def test_auto_resolution_does_resolve(init_full_x_full_y, monkeypatch):
    """Test that auto-resolution does resolve all conflicts"""
    # Patch cmdloop to avoid autoresolution's prompt
    monkeypatch.setattr("sys.__stdin__.isatty", lambda: True)

    name = "full_x_full_y"
    branch = "half_x_no_y_new_w"
    # If autoresolution was not successful, this to fail with a sys.exit without registering the
    # experiment
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} "
            "--enable-evc "
            "./black_box_with_y.py "
            "-x~uniform(0,10) "
            "-w~choices(['a','b'])"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )

    experiment = experiment_builder.load(name=branch)

    assert experiment.refers["adapter"].configuration == [
        {
            "of_type": "dimensionpriorchange",
            "name": "/x",
            "new_prior": "uniform(0, 10)",
            "old_prior": "uniform(-10, 10)",
        },
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/w", "type": "categorical", "value": None},
        },
    ]


def test_auto_resolution_with_fidelity(init_full_x_full_y, monkeypatch):
    """Test that auto-resolution does resolve all conflicts including new fidelity"""
    # Patch cmdloop to avoid autoresolution's prompt
    monkeypatch.setattr("sys.__stdin__.isatty", lambda: True)

    name = "full_x_full_y"
    branch = "half_x_no_y_new_w"
    # If autoresolution was not successful, this to fail with a sys.exit without registering the
    # experiment
    orion.core.cli.main(
        (
            "hunt --init-only -n {branch} --branch-from {name} "
            "--enable-evc "
            "./black_box_with_y.py "
            "-x~uniform(0,10) "
            "-w~fidelity(1,10)"
        )
        .format(name=name, branch=branch)
        .split(" ")
    )

    experiment = experiment_builder.load(name=branch)

    assert experiment.refers["adapter"].configuration == [
        {
            "of_type": "dimensionpriorchange",
            "name": "/x",
            "new_prior": "uniform(0, 10)",
            "old_prior": "uniform(-10, 10)",
        },
        {
            "of_type": "dimensiondeletion",
            "param": {"name": "/y", "type": "real", "value": 1},
        },
        {
            "of_type": "dimensionaddition",
            "param": {"name": "/w", "type": "fidelity", "value": 10},
        },
    ]


def test_init_w_version_from_parent_w_children(orionstate, monkeypatch, capsys):
    """Test that init of experiment from version with children fails."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    execute(
        "hunt --init-only -n experiment --config orion_config.yaml "
        "--enable-evc "
        "./black_box.py -x~normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment "
        "--enable-evc "
        "./black_box.py -x~normal(0,1) -y~+normal(0,1)"
    )

    execute(
        "hunt --init-only -n experiment -v 1 "
        "--enable-evc "
        "./black_box.py -x~normal(0,1) -y~+normal(0,1) -z~normal(0,1)",
        assert_code=1,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration is different and generates a branching event" in captured.err
    assert "Experiment name" in captured.err


def test_init_w_version_from_exp_wout_child(orionstate, monkeypatch):
    """Test that init of experiment from version without child works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    execute(
        "hunt --init-only -n experiment --config orion_config.yaml "
        "--enable-evc "
        "./black_box.py -x~normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment "
        "--enable-evc "
        "./black_box.py -x~normal(0,1) -y~+normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment -v 2 "
        "--enable-evc "
        "./black_box.py "
        "-x~normal(0,1) -y~+normal(0,1) -z~+normal(0,1)"
    )

    exp = setup_storage().fetch_experiments({"name": "experiment", "version": 3})
    assert len(list(exp))


def test_init_w_version_gt_max(orionstate, monkeypatch):
    """Test that init of experiment from version higher than max works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    execute(
        "hunt --init-only -n experiment --config orion_config.yaml "
        "--enable-evc "
        "./black_box.py -x~normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment "
        "--enable-evc "
        "./black_box.py -x~normal(0,1) -y~+normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment -v 2000 "
        "--enable-evc "
        "./black_box.py "
        "-x~normal(0,1) -y~+normal(0,1) -z~+normal(0,1)"
    )

    exp = setup_storage().fetch_experiments({"name": "experiment", "version": 3})
    assert len(list(exp))


def test_init_check_increment_w_children(orionstate, monkeypatch):
    """Test that incrementing version works with not same-named children."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    execute(
        "hunt --init-only -n experiment --config orion_config.yaml "
        "--enable-evc "
        "./black_box.py -x~normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment --branch-to experiment_2 "
        "--enable-evc "
        "./black_box.py "
        "-x~normal(0,1) -y~+normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment --enable-evc "
        "./black_box.py -x~normal(0,1) -z~+normal(0,1)"
    )

    exp = setup_storage().fetch_experiments({"name": "experiment", "version": 2})
    assert len(list(exp))


def test_branch_from_selected_version(orionstate, monkeypatch):
    """Test that branching from a version passed with `--version` works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    execute(
        "hunt --init-only -n experiment --config orion_config.yaml "
        "--enable-evc "
        "./black_box.py -x~normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment "
        "--enable-evc "
        "./black_box.py -x~normal(0,1) -y~+normal(0,1)"
    )
    execute(
        "hunt --init-only -n experiment --version 1 -b experiment_2 "
        "--enable-evc "
        "./black_box.py "
        "-x~normal(0,1) -z~+normal(0,1)"
    )

    storage = setup_storage()
    parent = storage.fetch_experiments({"name": "experiment", "version": 1})[0]
    exp = storage.fetch_experiments({"name": "experiment_2"})[0]
    assert exp["refers"]["parent_id"] == parent["_id"]
