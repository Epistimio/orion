#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.resolve_config`."""

import hashlib
import logging
import os
import shutil

import git
import pytest

import orion.core
import orion.core.io.resolve_config as resolve_config


@pytest.fixture
def force_is_exe(monkeypatch):
    """Mock resolve_config to recognize any string as an executable script."""

    def is_exe(path):
        return True

    monkeypatch.setattr(resolve_config, "is_exe", is_exe)


def test_fetch_env_vars():
    """Verify env vars are fetched properly"""
    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {"database": {}}

    db_name = "orion_test"

    os.environ["ORION_DB_NAME"] = db_name

    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {"database": {"name": "orion_test"}}

    db_type = "MongoDB"
    os.environ["ORION_DB_TYPE"] = db_type

    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {"database": {"name": db_name, "type": db_type}}


@pytest.mark.usefixtures("version_XYZ")
def test_fetch_metadata_orion_version():
    """Verify orion version"""
    metadata = resolve_config.fetch_metadata()
    assert metadata["orion_version"] == "XYZ"


def test_fetch_metadata_executable_users_script(script_path):
    """Verify executable user script with absolute path"""
    metadata = resolve_config.fetch_metadata(user_args=[script_path])
    assert metadata["user_script"] == os.path.abspath(script_path)


def test_fetch_metadata_non_executable_users_script():
    """Verify executable user script keeps given path"""
    script_path = "tests/functional/demo/orion_config.yaml"
    metadata = resolve_config.fetch_metadata(user_args=[script_path])
    assert metadata["user_script"] == script_path


def test_fetch_metadata_python_users_script(script_path):
    """Verify user script is correctly inferred if called with python"""
    metadata = resolve_config.fetch_metadata(
        user_args=["python", script_path, "some", "args"]
    )
    assert metadata["user_script"] == script_path


def test_fetch_metadata_not_existed_path():
    """Verfiy the raise of error when user_script path does not exist"""
    path = "dummy/path"
    with pytest.raises(OSError) as exc_info:
        resolve_config.fetch_metadata(user_args=[path])
    assert "The path specified for the script does not exist" in str(exc_info.value)


@pytest.mark.usefixtures()
def test_fetch_metadata_user_args(script_path):
    """Verify user args"""
    user_args = [os.path.abspath(script_path)] + list(map(str, range(10)))
    metadata = resolve_config.fetch_metadata(user_args=user_args)
    assert metadata["user_script"] == user_args[0]
    assert metadata["user_args"] == user_args


@pytest.mark.usefixtures("with_user_tsirif")
def test_fetch_metadata_user_tsirif():
    """Verify user name"""
    metadata = resolve_config.fetch_metadata()
    assert metadata["user"] == "tsirif"


def test_fetch_metadata():
    """Verify no additional data is stored in metadata"""
    metadata = resolve_config.fetch_metadata()
    len(metadata) == 4


def test_fetch_config_from_cmdargs():
    """Verify fetch_config returns empty dict on no config file path"""
    cmdargs = {
        "name": "test",
        "user": "me",
        "version": 1,
        "config": None,
        "exp_max_trials": "exp_max_trials",
        "worker_trials": "worker_trials",
        "exp_max_broken": "exp_max_broken",
        "working_dir": "working_dir",
        "max_trials": "max_trials",
        "heartbeat": "heartbeat",
        "worker_max_trials": "worker_max_trials",
        "worker_max_broken": "worker_max_broken",
        "max_idle_time": "max_idle_time",
        "reservation_timeout": "reservation_timeout",
        "interrupt_signal_code": "interrupt_signal_code",
        "user_script_config": "user_script_config",
        "manual_resolution": "manual_resolution",
        "non_monitored_arguments": "non_monitored_arguments",
        "ignore_code_changes": "ignore_code_changes",
        "auto_resolution": "auto_resolution",
        "branch_from": "branch_from",
        "algorithm_change": "algorithm_change",
        "code_change_type": "code_change_type",
        "cli_change_type": "cli_change_type",
        "branch_to": "branch_to",
        "config_change_type": "config_change_type",
    }

    config = resolve_config.fetch_config_from_cmdargs(cmdargs)

    assert config.pop("config", None) is None

    exp_config = config.pop("experiment")
    assert exp_config.pop("name") == "test"
    assert exp_config.pop("version") == 1
    assert exp_config.pop("user") == "me"
    assert exp_config.pop("max_trials") == "exp_max_trials"
    assert exp_config.pop("max_broken") == "exp_max_broken"
    assert exp_config.pop("working_dir") == "working_dir"

    assert exp_config == {}

    worker_config = config.pop("worker")
    assert worker_config.pop("heartbeat") == "heartbeat"
    assert worker_config.pop("max_trials") == "worker_max_trials"
    assert worker_config.pop("max_broken") == "worker_max_broken"
    assert worker_config.pop("max_idle_time") == "max_idle_time"
    assert worker_config.pop("reservation_timeout") == "reservation_timeout"
    assert worker_config.pop("interrupt_signal_code") == "interrupt_signal_code"
    assert worker_config.pop("user_script_config") == "user_script_config"

    assert worker_config == {}

    evc_config = config.pop("evc")
    assert evc_config.pop("manual_resolution") == "manual_resolution"
    assert evc_config.pop("non_monitored_arguments") == "non_monitored_arguments"
    assert evc_config.pop("ignore_code_changes") == "ignore_code_changes"
    assert evc_config.pop("auto_resolution") == "auto_resolution"
    assert evc_config.pop("branch_from") == "branch_from"
    assert evc_config.pop("algorithm_change") == "algorithm_change"
    assert evc_config.pop("code_change_type") == "code_change_type"
    assert evc_config.pop("cli_change_type") == "cli_change_type"
    assert evc_config.pop("branch_to") == "branch_to"
    assert evc_config.pop("config_change_type") == "config_change_type"

    assert evc_config == {}

    assert config == {}


@pytest.mark.parametrize(
    "argument",
    ["config", "user", "user_args", "name", "version", "branch_from", "branch_to"],
)
def test_fetch_config_from_cmdargs_no_empty(argument):
    """Verify fetch_config returns only defined arguments."""
    config = resolve_config.fetch_config_from_cmdargs({})
    assert config == {}

    config = resolve_config.fetch_config_from_cmdargs({argument: None})
    assert config == {}

    config = resolve_config.fetch_config_from_cmdargs({argument: False})
    assert config == {}

    config = resolve_config.fetch_config_from_cmdargs({argument: 1})

    if argument in ["name", "user", "version"]:
        assert config == {"experiment": {argument: 1}}
    elif argument in ["branch_from", "branch_to"]:
        assert config == {"evc": {argument: 1}}
    else:
        assert config == {argument: 1}


def test_fetch_config_no_hit():
    """Verify fetch_config returns empty dict on no config file path"""
    config = resolve_config.fetch_config({"config": ""})
    assert config == {}


def test_fetch_config(raw_config):
    """Verify fetch_config returns valid dictionary"""
    config = resolve_config.fetch_config({"config": raw_config})

    assert config.pop("storage") == {
        "database": {
            "host": "${FILE}",
            "name": "orion_test",
            "type": "pickleddb",
        }
    }

    assert config.pop("experiment") == {
        "max_trials": 100,
        "max_broken": 5,
        "name": "voila_voici",
        "algorithms": "random",
    }

    assert config == {}


def test_fetch_config_global_local_coherence(monkeypatch, config_file):
    """Verify fetch_config parses local config according to global config structure."""

    def mocked_config(file_object):
        return orion.core.config.to_dict()

    monkeypatch.setattr("yaml.safe_load", mocked_config)

    config = resolve_config.fetch_config({"config": config_file})

    # Test storage subconfig
    storage_config = config.pop("storage")
    database_config = storage_config.pop("database")
    assert storage_config.pop("type") == orion.core.config.storage.type
    assert storage_config == {}

    assert database_config.pop("host") == orion.core.config.storage.database.host
    assert database_config.pop("name") == orion.core.config.storage.database.name
    assert database_config.pop("port") == orion.core.config.storage.database.port
    assert database_config.pop("type") == orion.core.config.storage.database.type

    assert database_config == {}

    # Test experiment subconfig
    exp_config = config.pop("experiment")
    assert exp_config.pop("max_trials") == orion.core.config.experiment.max_trials
    assert exp_config.pop("max_broken") == orion.core.config.experiment.max_broken
    assert exp_config.pop("working_dir") == orion.core.config.experiment.working_dir
    assert exp_config.pop("algorithms") == orion.core.config.experiment.algorithms
    # TODO: Remove for v0.4
    assert exp_config.pop("strategy") == orion.core.config.experiment.strategy

    assert exp_config == {}

    # Test worker subconfig
    worker_config = config.pop("worker")
    assert worker_config.pop("n_workers") == orion.core.config.worker.n_workers
    assert worker_config.pop("pool_size") == orion.core.config.worker.pool_size
    assert worker_config.pop("executor") == orion.core.config.worker.executor
    assert (
        worker_config.pop("executor_configuration")
        == orion.core.config.worker.executor_configuration
    )
    assert worker_config.pop("heartbeat") == orion.core.config.worker.heartbeat
    assert worker_config.pop("max_trials") == orion.core.config.worker.max_trials
    assert worker_config.pop("max_broken") == orion.core.config.worker.max_broken
    assert worker_config.pop("max_idle_time") == orion.core.config.worker.max_idle_time
    assert (
        worker_config.pop("reservation_timeout")
        == orion.core.config.worker.reservation_timeout
    )
    assert worker_config.pop("idle_timeout") == orion.core.config.worker.idle_timeout
    assert (
        worker_config.pop("interrupt_signal_code")
        == orion.core.config.worker.interrupt_signal_code
    )
    assert (
        worker_config.pop("user_script_config")
        == orion.core.config.worker.user_script_config
    )

    assert worker_config == {}

    # Test evc subconfig
    evc_config = config.pop("evc")
    assert evc_config.pop("enable") is orion.core.config.evc.enable
    assert evc_config.pop("auto_resolution") == orion.core.config.evc.auto_resolution
    assert (
        evc_config.pop("manual_resolution") == orion.core.config.evc.manual_resolution
    )
    assert (
        evc_config.pop("non_monitored_arguments")
        == orion.core.config.evc.non_monitored_arguments
    )
    assert (
        evc_config.pop("ignore_code_changes")
        == orion.core.config.evc.ignore_code_changes
    )
    assert evc_config.pop("algorithm_change") == orion.core.config.evc.algorithm_change
    assert (
        evc_config.pop("orion_version_change")
        == orion.core.config.evc.orion_version_change
    )
    assert evc_config.pop("code_change_type") == orion.core.config.evc.code_change_type
    assert evc_config.pop("cli_change_type") == orion.core.config.evc.cli_change_type
    assert (
        evc_config.pop("config_change_type") == orion.core.config.evc.config_change_type
    )

    assert evc_config == {}

    # Test remaining config
    assert config.pop("debug") is False
    assert config.pop("frontends_uri") == []

    # Confirm that all fields were tested.
    assert config == {}


def test_fetch_config_dash(monkeypatch, config_file):
    """Verify fetch_config supports dash."""

    def mocked_config(file_object):
        return {"experiment": {"max-broken": 10, "algorithms": {"dont-change": "me"}}}

    monkeypatch.setattr("yaml.safe_load", mocked_config)

    config = resolve_config.fetch_config({"config": config_file})

    assert config == {
        "experiment": {"max_broken": 10, "algorithms": {"dont-change": "me"}}
    }


def test_fetch_config_underscore(monkeypatch, config_file):
    """Verify fetch_config supports underscore as well."""

    def mocked_config(file_object):
        return {"experiment": {"max_broken": 10, "algorithms": {"dont-change": "me"}}}

    monkeypatch.setattr("yaml.safe_load", mocked_config)

    config = resolve_config.fetch_config({"config": config_file})

    assert config == {
        "experiment": {"max_broken": 10, "algorithms": {"dont-change": "me"}}
    }


def test_fetch_config_deprecated_max_trials(monkeypatch, config_file):
    """Verify fetch_config will overwrite deprecated value if also properly defined."""

    def mocked_config(file_object):
        return {"experiment": {"max_trials": 10}, "max_trials": 20}

    monkeypatch.setattr("yaml.safe_load", mocked_config)

    config = resolve_config.fetch_config({"config": config_file})

    assert config == {"experiment": {"max_trials": 10}}


def test_fetch_config_deprecate_tricky_names(monkeypatch, config_file):
    """Verify fetch_config assigns values properly for the similar names."""

    def mocked_config(file_object):
        return {
            "experiment": {"worker_trials": "should_be_ignored"},
            "max_trials": "exp_max_trials",
            "max_broken": "exp_max_broken",
            "worker_trials": "worker_max_trials",
            "name": "exp_name",
        }

    monkeypatch.setattr("yaml.safe_load", mocked_config)

    config = resolve_config.fetch_config({"config": config_file})

    assert config == {
        "experiment": {
            "name": "exp_name",
            "max_trials": "exp_max_trials",
            "max_broken": "exp_max_broken",
        },
        "worker": {"max_trials": "worker_max_trials"},
    }


def test_merge_configs_update_two():
    """Ensure update on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"a": 3}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 3, "b": 2}


def test_merge_configs_update_three():
    """Ensure two updates on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"a": 3}
    c = {"b": 4}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {"a": 3, "b": 4}


def test_merge_configs_update_four():
    """Ensure three updates on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"a": 3}
    c = {"b": 4}
    d = {"a": 5, "b": 6}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {"a": 5, "b": 6}


def test_merge_configs_extend_two():
    """Ensure extension on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"c": 3}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": 2, "c": 3}


def test_merge_configs_extend_three():
    """Ensure two extensions on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"c": 3}
    c = {"d": 4}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_merge_configs_extend_four():
    """Ensure three extensions on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"c": 3}
    c = {"d": 4}
    d = {"e": 5}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def test_merge_configs_update_extend_two():
    """Ensure update and extension on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"b": 3, "c": 4}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": 3, "c": 4}


def test_merge_configs_update_extend_three():
    """Ensure two updates and extensions on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"b": 3, "c": 4}
    c = {"a": 5, "d": 6}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {"a": 5, "b": 3, "c": 4, "d": 6}


def test_merge_configs_update_extend_four():
    """Ensure three updates and extensions on first level is fine"""
    a = {"a": 1, "b": 2}
    b = {"b": 3, "c": 4}
    c = {"a": 5, "d": 6}
    d = {"d": 7, "e": 8}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {"a": 5, "b": 3, "c": 4, "d": 7, "e": 8}


def test_merge_sub_configs_update_two():
    """Ensure updating to second level is fine"""
    a = {"a": 1, "b": 2}
    b = {"b": {"c": 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": {"c": 3}}


def test_merge_sub_configs_sub_update_two():
    """Ensure updating on second level is fine"""
    a = {"a": 1, "b": {"c": 2}}
    b = {"b": {"c": 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": {"c": 3}}

    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"b": {"c": 4}}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": {"c": 4, "d": 3}}


def test_merge_sub_configs_sub_extend_two():
    """Ensure updating to third level from second level is fine"""
    a = {"a": 1, "b": {"c": 2}}
    b = {"d": {"e": 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": {"c": 2}, "d": {"e": 3}}

    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"b": {"e": {"f": 4}}}

    m = resolve_config.merge_configs(a, b)

    assert m == {"a": 1, "b": {"c": 2, "d": 3, "e": {"f": 4}}}


def test_merge_sub_configs_update_three():
    """Ensure updating twice to third level from second level is fine"""
    a = {"a": 1, "b": {"c": 2}}
    b = {"b": {"c": 3}}
    c = {"b": {"c": {"d": 4}}}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {"a": 1, "b": {"c": {"d": 4}}}

    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"b": {"c": 4}}
    c = {"b": {"c": {"e": 5}}}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {"a": 1, "b": {"c": {"e": 5}, "d": 3}}


def test_merge_matching_type_configs():
    """Test that configs with matching type are merged properly"""
    a = {"a": 1, "b": {"c": 2, "t": "match"}}
    b = {"b": {"c": 3, "d": 4, "t": "match"}}
    c = {"b": {"c": {"d": 4}, "e": 5, "t": "match"}}

    m = resolve_config.merge_configs(a, b, c, differentiators=["t"])

    assert m == {"a": 1, "b": {"c": {"d": 4}, "d": 4, "e": 5, "t": "match"}}


def test_merge_diff_type_configs():
    """Test that configs with diff type are not merged"""
    a = {"a": 1, "b": {"c": 2, "t": 1}}
    b = {"a": 1, "b": {"c": 3, "d": 4, "t": 2}}
    c = {"b": {"c": {"d": 4}, "e": 5, "t": 3}}

    m = resolve_config.merge_configs(a, b, differentiators=["t"])

    assert m == {"a": 1, "b": {"c": 3, "d": 4, "t": 2}}

    m = resolve_config.merge_configs(b, c, differentiators=["t"])
    assert m == {"a": 1, "b": {"c": {"d": 4}, "e": 5, "t": 3}}

    assert resolve_config.merge_configs(
        b, c, differentiators=["t"]
    ) == resolve_config.merge_configs(a, b, c, differentiators=["t"])


def test_merge_diff_type_sub_configs():
    """Test that configs with nested diff type are not merged"""
    a = {"a": 1, "b": {"c": 2, "t": 1, "d": {"t": 2, "e": 3}}}
    b = {"b": {"a": 3, "t": 1, "d": {"t": 2, "f": 4}}}

    m = resolve_config.merge_configs(a, b, differentiators=["t"])
    assert m == {"a": 1, "b": {"a": 3, "c": 2, "t": 1, "d": {"t": 2, "e": 3, "f": 4}}}

    c = {"b": {"a": 3, "t": 1, "d": {"t": 3, "f": 4}}}

    m = resolve_config.merge_configs(a, c, differentiators=["t"])
    assert m == {"a": 1, "b": {"a": 3, "c": 2, "t": 1, "d": {"t": 3, "f": 4}}}

    d = {"b": {"a": 3, "t": 2, "d": {"t": 2, "f": 4}}}

    m = resolve_config.merge_configs(a, d, differentiators=["t"])
    assert m == {"a": 1, "b": {"a": 3, "t": 2, "d": {"t": 2, "f": 4}}}


@pytest.fixture
def repo():
    """Create a dummy repo for the tests."""
    os.chdir("../")
    os.makedirs("dummy_orion")
    os.chdir("dummy_orion")
    repo = git.Repo.init(".")
    with open("README.md", "w+") as f:
        f.write("dummy content")
    repo.git.add("README.md")
    repo.index.commit("initial commit")
    repo.create_head("master")
    repo.git.checkout("master")
    yield repo
    os.chdir("../")
    shutil.rmtree("dummy_orion")
    os.chdir("orion")


@pytest.fixture
def invalid_repo():
    """Create a dummy invalid repo for the tests."""
    os.chdir("../")
    os.makedirs("dummy_orion")
    os.chdir("dummy_orion")
    repo = git.Repo.init(".")
    with open("README.md", "w+") as f:
        f.write("dummy content")
    # No commit, no branch, blank...
    yield repo
    os.chdir("../")
    shutil.rmtree("dummy_orion")
    os.chdir("orion")


def test_infer_version_on_invalid_head(invalid_repo, caplog):
    """Test that repo is ignored if repo has an invalid HEAD state."""

    with caplog.at_level(logging.WARNING):
        assert resolve_config.infer_versioning_metadata(".git") == {}

    assert "dummy_orion has an invalid HEAD." in caplog.text


def test_infer_versioning_metadata_on_clean_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the user's repo is clean:
    `is_dirty`, `active_branch` and `diff_sha`.
    """
    vcs = resolve_config.infer_versioning_metadata(".git")
    assert not vcs["is_dirty"]
    assert vcs["active_branch"] == "master"
    # the diff should be empty so the diff_sha should be equal to the diff sha of an empty string
    assert vcs["diff_sha"] == hashlib.sha256(b"").hexdigest()


def test_infer_versioning_metadata_on_dirty_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the uers's repo is dirty:
    `is_dirty`, `HEAD_sha`, `active_branch` and `diff_sha`.
    """
    existing_metadata = {}
    existing_metadata["user_script"] = ".git"
    vcs = resolve_config.infer_versioning_metadata(".git")
    repo.create_head("feature")
    repo.git.checkout("feature")
    with open("README.md", "w+") as f:
        f.write("dummy dummy content")
    vcs = resolve_config.infer_versioning_metadata(".git")
    assert vcs["is_dirty"]
    assert vcs["active_branch"] == "feature"
    assert vcs["diff_sha"] != hashlib.sha256(b"").hexdigest()
    repo.git.add("README.md")
    commit = repo.index.commit("Added dummy_file")
    vcs = resolve_config.infer_versioning_metadata(".git")
    assert not vcs["is_dirty"]
    assert vcs["HEAD_sha"] == commit.hexsha
    assert vcs["diff_sha"] == hashlib.sha256(b"").hexdigest()


def test_fetch_user_repo_on_non_repo(caplog):
    """Test if `fetch_user_repo` logs a warning when user's script is not a git repo."""
    with caplog.at_level(logging.WARNING):
        resolve_config.fetch_user_repo(".")
    messages = [rec.message for rec in caplog.records]
    assert len(messages) == 1
    assert f"Script {os.getcwd()} is not in a git repository" in messages[0]


def test_infer_versioning_metadata_on_detached_head(repo):
    """Test in the case of a detached head."""
    with open("README.md", "w+") as f:
        f.write("dummy contentt")
    repo.git.add("README.md")
    repo.index.commit("2nd commit")
    existing_metadata = {}
    existing_metadata["user_script"] = ".git"
    repo.head.reference = repo.commit("HEAD~1")
    assert repo.head.is_detached
    vcs = resolve_config.infer_versioning_metadata(".git")
    assert vcs["active_branch"] is None
