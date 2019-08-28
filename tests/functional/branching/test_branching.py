#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for branching."""

import os

import pytest

import orion.core.cli
from orion.core.io.evc_builder import EVCBuilder
from orion.core.worker.experiment import ExperimentView


@pytest.fixture
def init_full_x(clean_db, monkeypatch):
    """Init original experiment"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    name = "full_x"
    orion.core.cli.main(("init_only -n {name} --config orion_config.yaml ./black_box.py "
                         "-x~uniform(-10,10)").format(name=name).split(" "))
    orion.core.cli.main("insert -n {name} script -x=0".format(name=name).split(" "))


@pytest.fixture
def init_full_x_full_y(init_full_x):
    """Add y dimension to original"""
    name = "full_x"
    branch = "full_x_full_y"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_y.py "
         "-x~uniform(-10,10) "
         "-y~+uniform(-10,10,default_value=1)").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=1 -y=1".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-1 -y=1".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=1 -y=-1".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-1 -y=-1".format(name=branch).split(" "))


@pytest.fixture
def init_half_x_full_y(init_full_x_full_y):
    """Change x's prior to full x and full y experiment"""
    name = "full_x_full_y"
    branch = "half_x_full_y"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_y.py "
         "-x~+uniform(0,10) "
         "-y~uniform(-10,10,default_value=1)").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=2 -y=2".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=2 -y=-2".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_half_y(init_full_x_full_y):
    """Change y's prior to full x and full y experiment"""
    name = "full_x_full_y"
    branch = "full_x_half_y"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_y.py "
         "-x~uniform(-10,10) "
         "-y~+uniform(0,10,default_value=1)").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=3 -y=3".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-3 -y=3".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_rename_y_z(init_full_x_full_y):
    """Rename y from full x full y to z"""
    name = "full_x_full_y"
    branch = "full_x_rename_y_z"
    orion.core.cli.main(("init_only -n {name} --branch {branch} ./black_box_with_z.py "
                         "-x~uniform(-10,10) -y~>z -z~uniform(-10,10,default_value=1)"
                         ).format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=4 -z=4".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-4 -z=4".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=4 -z=-4".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-4 -z=-4".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_rename_half_y_half_z(init_full_x_half_y):
    """Rename y from full x half y to z"""
    name = "full_x_half_y"
    branch = "full_x_rename_half_y_half_z"
    orion.core.cli.main(("init_only -n {name} --branch {branch} ./black_box_with_z.py "
                         "-x~uniform(-10,10) -y~>z -z~uniform(0,10,default_value=1)"
                         ).format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=5 -z=5".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-5 -z=5".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_rename_half_y_full_z(init_full_x_half_y):
    """Rename y from full x half y to full z (rename + changed prior)"""
    name = "full_x_half_y"
    branch = "full_x_rename_half_y_full_z"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_z.py "
         "-x~uniform(-10,10) -y~>z "
         "-z~+uniform(-10,10,default_value=1)").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=6 -z=6".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-6 -z=6".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=6 -z=-6".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-6 -z=-6".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_remove_y(init_full_x_full_y):
    """Remove y from full x full y"""
    name = "full_x_full_y"
    branch = "full_x_remove_y"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box.py "
         "-x~uniform(-10,10) -y~-").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=7".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-7".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_remove_z(init_full_x_rename_y_z):
    """Remove z from full x full z"""
    name = "full_x_rename_y_z"
    branch = "full_x_remove_z"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box.py "
         "-x~uniform(-10,10) -z~-").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=8".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-8".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_remove_z_default_4(init_full_x_rename_y_z):
    """Remove z from full x full z and give a default value of 4"""
    name = "full_x_rename_y_z"
    branch = "full_x_remove_z_default_4"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box.py "
         "-x~uniform(-10,10) -z~-4").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=9".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-9".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_new_algo(init_full_x):
    """Remove z from full x full z and give a default value of 4"""
    name = "full_x"
    branch = "full_x_new_algo"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} --algorithm-change --config new_algo_config.yaml "
         "./black_box.py -x~uniform(-10,10)").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=1.1".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-1.1".format(name=branch).split(" "))


@pytest.fixture
def init_full_x_new_cli(init_full_x):
    """Remove z from full x full z and give a default value of 4"""
    name = "full_x"
    branch = "full_x_new_cli"
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} --cli-change-type noeffect ./black_box_new.py "
         "-x~uniform(-10,10) --a-new argument").format(name=name, branch=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=1.2".format(name=branch).split(" "))
    orion.core.cli.main("insert -n {name} script -x=-1.2".format(name=branch).split(" "))


@pytest.fixture
def init_entire(init_half_x_full_y,  # 1.1.1
                init_full_x_rename_half_y_half_z,  # 1.1.2.1
                init_full_x_rename_half_y_full_z,  # 1.1.2.2
                init_full_x_remove_y,  # 1.1.4
                init_full_x_remove_z,  # 1.1.3.1
                init_full_x_remove_z_default_4):  # 1.1.3.2
    """Initialize all experiments"""
    pass


def get_name_value_pairs(trials):
    """Turn parameters into pairs for easy comparisions"""
    pairs = []
    for trial in trials:
        pairs.append([])
        for param in trial.params:
            pairs[-1].append((param.name, param.value))

        pairs[-1] = tuple(pairs[-1])

    return tuple(pairs)


def test_init(init_full_x, create_db_instance):
    """Test if original experiment contains trial 0"""
    experiment = ExperimentView('full_x')
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 0), ), )


def test_full_x_full_y(init_full_x_full_y, create_db_instance):
    """Test if full x full y is properly initialized and can fetch original trial"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_full_y'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 1), ('/y', 1)),
                     (('/x', -1), ('/y', 1)),
                     (('/x', 1), ('/y', -1)),
                     (('/x', -1), ('/y', -1)))

    # pytest.set_trace()
    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/y', 1)),
                     (('/x', 1), ('/y', 1)),
                     (('/x', -1), ('/y', 1)),
                     (('/x', 1), ('/y', -1)),
                     (('/x', -1), ('/y', -1)))


def test_half_x_full_y(init_half_x_full_y, create_db_instance):
    """Test if half x full y is properly initialized and can fetch from its 2 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'half_x_full_y'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 2), ('/y', 2)),
                     (('/x', 2), ('/y', -2)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/y', 1)),
                     (('/x', 1), ('/y', 1)),
                     (('/x', 1), ('/y', -1)),
                     (('/x', 2), ('/y', 2)),
                     (('/x', 2), ('/y', -2)))


def test_full_x_half_y(init_full_x_half_y, create_db_instance):
    """Test if full x half y is properly initialized and can fetch from its 2 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_half_y'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 3), ('/y', 3)),
                     (('/x', -3), ('/y', 3)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/y', 1)),
                     (('/x', 1), ('/y', 1)),
                     (('/x', -1), ('/y', 1)),
                     (('/x', 3), ('/y', 3)),
                     (('/x', -3), ('/y', 3)))


def test_full_x_rename_y_z(init_full_x_rename_y_z, create_db_instance):
    """Test if full x full z is properly initialized and can fetch from its 2 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_rename_y_z'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 4), ('/z', 4)),
                     (('/x', -4), ('/z', 4)),
                     (('/x', 4), ('/z', -4)),
                     (('/x', -4), ('/z', -4)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/z', 1)),
                     (('/x', 1), ('/z', 1)),
                     (('/x', -1), ('/z', 1)),
                     (('/x', 1), ('/z', -1)),
                     (('/x', -1), ('/z', -1)),
                     (('/x', 4), ('/z', 4)),
                     (('/x', -4), ('/z', 4)),
                     (('/x', 4), ('/z', -4)),
                     (('/x', -4), ('/z', -4)))


def test_full_x_rename_half_y_half_z(init_full_x_rename_half_y_half_z, create_db_instance):
    """Test if full x half z is properly initialized and can fetch from its 3 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_rename_half_y_half_z'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 5), ('/z', 5)),
                     (('/x', -5), ('/z', 5)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/z', 1)),
                     (('/x', 1), ('/z', 1)),
                     (('/x', -1), ('/z', 1)),
                     (('/x', 3), ('/z', 3)),
                     (('/x', -3), ('/z', 3)),
                     (('/x', 5), ('/z', 5)),
                     (('/x', -5), ('/z', 5)))


def test_full_x_rename_half_y_full_z(init_full_x_rename_half_y_full_z, create_db_instance):
    """Test if full x half->full z is properly initialized and can fetch from its 3 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_rename_half_y_full_z'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 6), ('/z', 6)),
                     (('/x', -6), ('/z', 6)),
                     (('/x', 6), ('/z', -6)),
                     (('/x', -6), ('/z', -6)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ('/z', 1)),
                     (('/x', 1), ('/z', 1)),
                     (('/x', -1), ('/z', 1)),
                     (('/x', 3), ('/z', 3)),
                     (('/x', -3), ('/z', 3)),
                     (('/x', 6), ('/z', 6)),
                     (('/x', -6), ('/z', 6)),
                     (('/x', 6), ('/z', -6)),
                     (('/x', -6), ('/z', -6)))


def test_full_x_remove_y(init_full_x_remove_y, create_db_instance):
    """Test if full x removed y is properly initialized and can fetch from its 2 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_remove_y'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 7), ), (('/x', -7), ))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert pairs == ((('/x', 0), ), (('/x', 1), ), (('/x', -1), ), (('/x', 7), ), (('/x', -7), ))


def test_full_x_remove_z(init_full_x_remove_z, create_db_instance):
    """Test if full x removed z is properly initialized and can fetch from 2 of its 3 parents"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_remove_z'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 8), ), (('/x', -8), ))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    # Note that full_x_rename_y_z are filtered out because default_value=1
    assert pairs == ((('/x', 0), ), (('/x', 1), ), (('/x', -1), ), (('/x', 8), ), (('/x', -8), ))


def test_full_x_remove_z_default_4(init_full_x_remove_z_default_4, create_db_instance):
    """Test if full x removed z  (default 4) is properly initialized and can fetch
    from 1 of its 3 parents
    """
    experiment = EVCBuilder().build_view_from({'name': 'full_x_remove_z_default_4'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 9), ), (('/x', -9), ))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    # Note that full_x and full_x_full_y are filtered out because default_value=4
    assert pairs == ((('/x', 4), ), (('/x', -4), ), (('/x', 9), ), (('/x', -9), ))


def test_entire_full_x_full_y(init_entire, create_db_instance):
    """Test if full x full y can fetch from its parent and all children"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_full_y'})
    pairs = get_name_value_pairs(experiment.fetch_trials())
    assert pairs == ((('/x', 1), ('/y', 1)),
                     (('/x', -1), ('/y', 1)),
                     (('/x', 1), ('/y', -1)),
                     (('/x', -1), ('/y', -1)))

    pairs = get_name_value_pairs(experiment.fetch_trials(with_evc_tree=True))
    assert set(pairs) == set(((('/x', 0), ('/y', 1)),
                              # full_x_full_y
                              (('/x', 1), ('/y', 1)),
                              (('/x', -1), ('/y', 1)),
                              (('/x', 1), ('/y', -1)),
                              (('/x', -1), ('/y', -1)),
                              # half_x_full_y
                              (('/x', 2), ('/y', 2)),
                              (('/x', 2), ('/y', -2)),
                              # full_x_half_y
                              (('/x', 3), ('/y', 3)),
                              (('/x', -3), ('/y', 3)),
                              # full_x_rename_y_z
                              (('/x', 4), ('/y', 4)),
                              (('/x', -4), ('/y', 4)),
                              (('/x', 4), ('/y', -4)),
                              (('/x', -4), ('/y', -4)),
                              # full_x_rename_half_y_half_z
                              (('/x', 5), ('/y', 5)),
                              (('/x', -5), ('/y', 5)),
                              # full_x_rename_half_y_full_z
                              (('/x', 6), ('/y', 6)),
                              (('/x', -6), ('/y', 6)),
                              # full_x_remove_y
                              (('/x', 7), ('/y', 1)),
                              (('/x', -7), ('/y', 1)),
                              # full_x_remove_z
                              (('/x', 8), ('/y', 1)),
                              (('/x', -8), ('/y', 1)),
                              # full_x_remove_z_default_4
                              (('/x', 9), ('/y', 4)),
                              (('/x', -9), ('/y', 4))))


def test_run_entire_full_x_full_y(init_entire, create_db_instance):
    """Test if branched experiment can be executed without triggering a branching event again"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_full_y'})
    assert len(experiment.fetch_trials(with_evc_tree=True)) == 23
    assert len(experiment.fetch_trials()) == 4

    orion.core.cli.main(("-vv hunt --max-trials 20 --pool-size 1 -n full_x_full_y "
                         "./black_box_with_y.py "
                         "-x~uniform(-10,10) "
                         "-y~uniform(-10,10,default_value=1)").split(" "))

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 39
    assert len(experiment.fetch_trials()) == 20


def test_run_entire_full_x_full_y_no_args(init_entire, create_db_instance):
    """Test if branched experiment can be executed without script arguments"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_full_y'})
    assert len(experiment.fetch_trials(with_evc_tree=True)) == 23
    assert len(experiment.fetch_trials()) == 4

    orion.core.cli.main(("-vv hunt --max-trials 20 --pool-size 1 -n full_x_full_y").split(" "))

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 39
    assert len(experiment.fetch_trials()) == 20


def test_new_algo(init_full_x_new_algo):
    """Test that new algo conflict is automatically resolved"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_new_algo'})
    assert len(experiment.fetch_trials(with_evc_tree=True)) == 3
    assert len(experiment.fetch_trials()) == 2

    orion.core.cli.main(("-vv hunt --max-trials 20 --pool-size 1 -n full_x_new_algo").split(" "))

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 21
    assert len(experiment.fetch_trials()) == 20


def test_new_algo_not_resolved(init_full_x):
    """Test that new algo conflict is not automatically resolved"""
    name = "full_x"
    branch = "full_x_new_algo"
    with pytest.raises(ValueError) as exc:
        orion.core.cli.main(
            ("init_only -n {name} --branch {branch} --config new_algo_config.yaml "
             "--manual-resolution ./black_box.py -x~uniform(-10,10)")
            .format(name=name, branch=branch).split(" "))
    assert "Configuration is different and generates a branching event" in str(exc.value)


def test_new_cli(init_full_x_new_cli):
    """Test that new cli conflict is automatically resolved"""
    experiment = EVCBuilder().build_view_from({'name': 'full_x_new_cli'})
    assert len(experiment.fetch_trials(with_evc_tree=True)) == 3
    assert len(experiment.fetch_trials()) == 2

    orion.core.cli.main(("-vv hunt --max-trials 20 --pool-size 1 -n full_x_new_cli").split(" "))

    assert len(experiment.fetch_trials(with_evc_tree=True)) == 21
    assert len(experiment.fetch_trials()) == 20


def test_auto_resolution_does_resolve(init_full_x_full_y, monkeypatch):
    """Test that auto-resolution does resolve all conflicts"""
    # Patch cmdloop to avoid autoresolution's prompt
    monkeypatch.setattr('sys.__stdin__.isatty', lambda: True)

    name = "full_x_full_y"
    branch = "half_x_no_y_new_w"
    # If autoresolution was not succesfull, this to fail with a sys.exit without registering the
    # experiment
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_y.py "
         "-x~uniform(0,10) "
         "-w~choices(['a','b'])").format(name=name, branch=branch).split(" "))


def test_auto_resolution_with_fidelity(init_full_x_full_y, monkeypatch):
    """Test that auto-resolution does resolve all conflicts including new fidelity"""
    # Patch cmdloop to avoid autoresolution's prompt
    monkeypatch.setattr('sys.__stdin__.isatty', lambda: True)

    name = "full_x_full_y"
    branch = "half_x_no_y_new_w"
    # If autoresolution was not succesfull, this to fail with a sys.exit without registering the
    # experiment
    orion.core.cli.main(
        ("init_only -n {name} --branch {branch} ./black_box_with_y.py "
         "-x~uniform(0,10) "
         "-w~fidelity(1,10)").format(name=name, branch=branch).split(" "))


def test_init_w_version_from_parent_w_children(clean_db, monkeypatch):
    """Test that init of experiment from version with children fails."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1) "
                        "-y~+normal(0,1)".split(" "))

    with pytest.raises(ValueError) as exc:
        orion.core.cli.main("init_only -n experiment -v 1 ./black_box.py "
                            "-x~normal(0,1) -y~+normal(0,1) -z~normal(0,1)".split(" "))

    assert "Experiment name" in str(exc.value)


def test_init_w_version_from_exp_wout_child(clean_db, monkeypatch, database):
    """Test that init of experiment from version without child works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1) "
                        "-y~+normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment -v 2 ./black_box.py "
                        "-x~normal(0,1) -y~+normal(0,1) -z~+normal(0,1)".split(" "))

    exp = database.experiments.find({'name': 'experiment', 'version': 3})
    assert len(list(exp))


def test_init_w_version_gt_max(clean_db, monkeypatch, database):
    """Test that init of experiment from version higher than max works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1) "
                        "-y~+normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment -v 2000 ./black_box.py "
                        "-x~normal(0,1) -y~+normal(0,1) -z~+normal(0,1)".split(" "))

    exp = database.experiments.find({'name': 'experiment', 'version': 3})
    assert len(list(exp))


def test_init_check_increment_w_children(clean_db, monkeypatch, database):
    """Test that incrementing version works with not same-named children."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment --branch experiment_2 ./black_box.py "
                        "-x~normal(0,1) -y~+normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment ./black_box.py "
                        "-x~normal(0,1) -z~+normal(0,1)".split(" "))

    exp = database.experiments.find({'name': 'experiment', 'version': 2})
    assert len(list(exp))


def test_branch_from_selected_version(clean_db, monkeypatch, database):
    """Test that branching from a version passed with `--version` works."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1)".split(" "))
    orion.core.cli.main("init_only -n experiment ./black_box.py -x~normal(0,1) -y~+normal(0,1)"
                        .split(" "))
    orion.core.cli.main("init_only -n experiment --version 1 -b experiment_2 ./black_box.py "
                        "-x~normal(0,1) -z~+normal(0,1)".split(" "))

    parent = database.experiments.find({'name': 'experiment', 'version': 1})[0]
    exp = database.experiments.find({'name': 'experiment_2'})[0]
    assert exp['refers']['parent_id'] == parent['_id']
