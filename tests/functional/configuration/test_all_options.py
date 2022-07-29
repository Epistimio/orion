"""Perform functional tests merge of configuration levels"""
import copy
import datetime
import os
import random
import shutil
import tempfile
from contextlib import contextmanager

import pytest
import yaml

import orion.core
import orion.core.cli
import orion.core.cli.hunt
import orion.core.evc.conflicts
import orion.core.io.resolve_config
from orion.client import get_experiment
from orion.core.io import experiment_builder
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage
from orion.storage.legacy import Legacy
from orion.testing.state import OrionState

script = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "..", "demo", "black_box.py"
)


def with_storage_fork(func):
    """Copy PickledDB to a tmp address and work in the tmp path within the func execution.

    Functions decorated with this decorator should only be called after the storage has been
    initialized.
    """

    def call(*args, **kwargs):

        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:

            storage = setup_storage()
            old_path = storage._db.host

            orion.core.config.storage.database.host = tmp_file.name
            shutil.copyfile(old_path, tmp_file.name)

            rval = func(*args, **kwargs)

            orion.core.config.storage.database.host = old_path

        return rval

    return call


class ConfigurationTestSuite:
    """Test suite for the configuration groups"""

    database = {}

    default_storage = {
        "type": "legacy",
        "database": {"type": "pickleddb", "host": "${file}_experiment.pkl"},
    }

    @contextmanager
    def setup_global_config(self, tmp_path):
        """Setup temporary yaml file for the global configuration"""
        with OrionState(storage=self.default_storage) as cfg:
            conf_file = tmp_path / "config.yaml"

            if "storage" not in self.config:
                self.config["storage"] = self.default_storage

            config_str = yaml.dump(self.config)
            config_str = config_str.replace("${tmp_path}", str(tmp_path))
            config_str = config_str.replace("${file}", str(cfg.tempfile_path))

            conf_file.write_text(config_str)
            conf_files = orion.core.DEF_CONFIG_FILES_PATHS
            orion.core.DEF_CONFIG_FILES_PATHS = [conf_file]

            orion.core.config = orion.core.build_config()

            try:
                yield conf_file
            finally:
                try:
                    os.remove(orion.core.config.storage.database.host)
                except:
                    pass

                orion.core.DEF_CONFIG_FILES_PATHS = conf_files
                orion.core.config = orion.core.build_config()

    @contextmanager
    def setup_env_var_config(self, tmp_path):
        """Setup environment variables with temporary values"""
        with self.setup_global_config(tmp_path):
            tmp = {}
            for key, value in self.env_vars.items():
                if isinstance(value, str):
                    value = value.replace("${tmp_path}", str(tmp_path))

                tmp[key] = os.environ.pop(key, None)
                os.environ[key] = str(value)
            try:
                yield
            finally:
                for key, value in tmp.items():
                    if value:
                        os.environ[key] = str(value)
                    else:
                        del os.environ[key]

    @contextmanager
    def setup_db_config(self, tmp_path):
        """Setup database with temporary data"""
        with self.setup_env_var_config(tmp_path):
            storage = setup_storage()
            storage.create_experiment(self.database)
            yield storage

    @contextmanager
    def setup_local_config(self, tmp_path):
        """Setup local configuration on top"""
        with self.setup_db_config(tmp_path):
            conf_file = tmp_path / "local.yaml"

            config_str = yaml.dump(self.local)
            config_str = config_str.replace("${tmp_path}", str(tmp_path))

            conf_file.write_text(config_str)
            yield conf_file

    @contextmanager
    def setup_cmd_args_config(self, tmp_path):
        """Setup cmd args configuration... do nothing actually?"""
        with self.setup_local_config(tmp_path) as conf_file:
            yield conf_file

    def test_global_config(self, tmp_path, monkeypatch):
        """Test that global configuration is set properly based on global yaml"""

        self.sanity_check()
        with self.setup_global_config(tmp_path):
            self.check_global_config(tmp_path, monkeypatch)

    def test_env_var_config(self, tmp_path, monkeypatch):
        """Test that env vars are set properly in global config"""

        self.sanity_check()
        with self.setup_env_var_config(tmp_path):
            self.check_env_var_config(tmp_path, monkeypatch)

    @pytest.mark.usefixtures(
        "with_user_userxyz", "version_XYZ", "mock_infer_versioning_metadata"
    )
    def test_db_config(self, tmp_path):
        """Test that exp config in db overrides global config"""

        self.sanity_check()
        with self.setup_db_config(tmp_path):
            self.check_db_config()

    @pytest.mark.usefixtures("with_user_userxyz", "version_XYZ")
    def test_local_config(self, tmp_path, monkeypatch):
        """Test that local config overrides db/global config"""

        self.sanity_check()
        with self.setup_local_config(tmp_path) as conf_file:
            self.check_local_config(tmp_path, conf_file, monkeypatch)

    @pytest.mark.usefixtures("with_user_userxyz", "version_XYZ")
    def test_cmd_args_config(self, tmp_path, monkeypatch):
        """Test that cmd_args config overrides local config"""

        self.sanity_check()
        with self.setup_cmd_args_config(tmp_path) as conf_file:
            self.check_cmd_args_config(tmp_path, conf_file, monkeypatch)


class TestStorage(ConfigurationTestSuite):
    """Test suite for storage configuration"""

    config = {
        "storage": {
            "type": "legacy",
            "database": {
                "name": "test_name",
                "type": "pickleddb",
                "host": "${tmp_path}/here.pkl",
                "port": 101,
            },
        }
    }

    env_vars = {
        "ORION_STORAGE_TYPE": "legacy",
        "ORION_DB_NAME": "test_env_var_name",
        "ORION_DB_TYPE": "pickleddb",
        "ORION_DB_ADDRESS": "${tmp_path}/there.pkl",
        "ORION_DB_PORT": "103",
    }

    local = {
        "storage": {
            "type": "legacy",
            "database": {"type": "pickleddb", "host": "${tmp_path}/local.pkl"},
        }
    }

    def sanity_check(self):
        """Check that defaults are different than testing configuration"""
        assert orion.core.config.storage.to_dict() != self.config["storage"]

    def check_global_config(self, tmp_path, monkeypatch):
        """Check that global configuration is set properly"""

        storage_config = copy.deepcopy(self.config["storage"])
        storage_config["database"]["host"] = storage_config["database"]["host"].replace(
            "${tmp_path}", str(tmp_path)
        )
        assert orion.core.config.storage.to_dict() == storage_config

        # Build storage
        storage = setup_storage()
        assert len(storage.fetch_experiments({"name": "test"})) == 0

        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        # if hunt worked it should insert its experiment
        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == str(tmp_path / "here.pkl")

    def check_env_var_config(self, tmp_path, monkeypatch):
        """Check that env vars overrides global configuration"""

        assert orion.core.config.storage.to_dict() == {
            "type": self.env_vars["ORION_STORAGE_TYPE"],
            "database": {
                "name": self.env_vars["ORION_DB_NAME"],
                "type": self.env_vars["ORION_DB_TYPE"],
                "host": self.env_vars["ORION_DB_ADDRESS"].replace(
                    "${tmp_path}", str(tmp_path)
                ),
                "port": int(self.env_vars["ORION_DB_PORT"]),
            },
        }

        # Build storage
        storage = setup_storage()
        assert len(storage.fetch_experiments({"name": "test"})) == 0

        # Make sure hunt is picking up the right database
        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        # if hunt worked it should insert its experiment
        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == self.env_vars["ORION_DB_ADDRESS"].replace(
            "${tmp_path}", str(tmp_path)
        )

    def check_db_config(self):
        """No Storage config in DB, no test"""

    def check_local_config(self, tmp_path, conf_file, monkeypatch):
        """Check that local configuration overrides global/envvars configuration"""

        assert orion.core.config.storage.to_dict() == {
            "type": self.env_vars["ORION_STORAGE_TYPE"],
            "database": {
                "name": self.env_vars["ORION_DB_NAME"],
                "type": self.env_vars["ORION_DB_TYPE"],
                "host": self.env_vars["ORION_DB_ADDRESS"].replace(
                    "${tmp_path}", str(tmp_path)
                ),
                "port": int(self.env_vars["ORION_DB_PORT"]),
            },
        }

        # Build storage with local config
        cmd_config = experiment_builder.get_cmd_config(dict(config=open(conf_file)))
        builder = experiment_builder.ExperimentBuilder(cmd_config["storage"])
        storage = builder.storage

        assert len(storage.fetch_experiments({"name": "test"})) == 0

        # Make sure hunt is picking up the right database
        command = f"hunt --exp-max-trials 0 -n test -c {conf_file} python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        # if hunt worked it should insert its experiment
        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == str(tmp_path / "local.pkl")

    def check_cmd_args_config(self, tmp_path, conf_file, monkeypatch):
        """No Storage config in cmdline, no test"""


class TestDatabaseDeprecated(ConfigurationTestSuite):
    """Test suite for deprecated database configuration."""

    config = {
        "database": {
            "name": "test_name",
            "type": "pickleddb",
            "host": "${tmp_path}/dbhere.pkl",
            "port": 101,
        }
    }

    env_vars = {
        "ORION_DB_NAME": "test_env_var_name",
        "ORION_DB_TYPE": "pickleddb",
        "ORION_DB_ADDRESS": "${tmp_path}/dbthere.pkl",
        "ORION_DB_PORT": "103",
    }

    local = {"database": {"type": "pickleddb", "host": "${tmp_path}/dblocal.pkl"}}

    def sanity_check(self):
        """Check that defaults are different than testing configuration"""
        assert orion.core.config.database.to_dict() != self.config["database"]

    def check_global_config(self, tmp_path, monkeypatch):
        """Check that global configuration is set properly"""

        database = copy.deepcopy(self.config["database"])
        database["host"] = database["host"].replace("${tmp_path}", str(tmp_path))
        assert orion.core.config.database.to_dict() == database

        storage = setup_storage()
        assert len(storage.fetch_experiments({"name": "test"})) == 0

        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == str(tmp_path / "dbhere.pkl")

    def check_env_var_config(self, tmp_path, monkeypatch):
        """Check that env vars overrides global configuration"""

        assert orion.core.config.database.to_dict() == {
            "name": self.env_vars["ORION_DB_NAME"],
            "type": self.env_vars["ORION_DB_TYPE"],
            "host": self.env_vars["ORION_DB_ADDRESS"].replace(
                "${tmp_path}", str(tmp_path)
            ),
            "port": int(self.env_vars["ORION_DB_PORT"]),
        }

        storage = setup_storage()
        assert len(storage.fetch_experiments({"name": "test"})) == 0

        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == self.env_vars["ORION_DB_ADDRESS"].replace(
            "${tmp_path}", str(tmp_path)
        )

    def check_db_config(self):
        """No Storage config in DB, no test"""

    def check_local_config(self, tmp_path, conf_file, monkeypatch):
        """Check that local configuration overrides global/envvars configuration"""

        assert orion.core.config.database.to_dict() == {
            "name": self.env_vars["ORION_DB_NAME"],
            "type": self.env_vars["ORION_DB_TYPE"],
            "host": self.env_vars["ORION_DB_ADDRESS"].replace(
                "${tmp_path}", str(tmp_path)
            ),
            "port": int(self.env_vars["ORION_DB_PORT"]),
        }

        cmd_config = experiment_builder.get_cmd_config(dict(config=open(conf_file)))
        builder = experiment_builder.ExperimentBuilder(cmd_config["storage"])
        storage = builder.storage

        assert len(storage.fetch_experiments({"name": "test"})) == 0

        # Make sure hunt is picking up the right database
        command = f"hunt --exp-max-trials 0 -n test -c {conf_file} python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        # if hunt worked it should insert its experiment
        assert len(storage.fetch_experiments({"name": "test"})) == 1

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)
        assert storage._db.host == str(tmp_path / "dblocal.pkl")

    def check_cmd_args_config(self, tmp_path, conf_file, monkeypatch):
        """No Storage config in cmdline, no test"""


class TestExperimentConfig(ConfigurationTestSuite):
    """Test suite for experiment configuration"""

    config = {
        "experiment": {
            "max_trials": 10,
            "max_broken": 5,
            "working_dir": "here",
            "worker_trials": 5,
            "algorithms": {"aa": {"b": "c", "d": {"e": "f"}}},
        }
    }

    env_vars = {
        "ORION_EXP_MAX_TRIALS": 20,
        "ORION_EXP_MAX_BROKEN": 12,
        "ORION_WORKING_DIR": "over_there",
    }

    database = {
        "name": "test-name",
        "version": 1,
        "max_trials": 75,
        "max_broken": 16,
        "working_dir": "in_db?",
        "algorithms": {"ab": {"d": "i", "f": "g"}},
        "space": {"/x": "uniform(0, 1)"},
        "metadata": {
            "VCS": {
                "HEAD_sha": "test",
                "active_branch": None,
                "diff_sha": "diff",
                "is_dirty": False,
                "type": "git",
            },
            "datetime": datetime.datetime.utcnow(),
            "orion_version": "XYZ",
            "parser": {
                "cmd_priors": [["/x", "uniform(0, 1)"]],
                "config_file_data": {},
                "config_prefix": "config",
                "converter": None,
                "file_config_path": None,
                "file_priors": [],
                "parser": {
                    "arguments": [
                        ["_pos_0", "python"],
                        ["_pos_1", script],
                        ["x", "orion~uniform(0, 1)"],
                    ],
                    "keys": [["_pos_0", "_pos_0"], ["_pos_1", "_pos_1"], ["x", "-x"]],
                    "template": ["{_pos_0}", "{_pos_1}", "-x", "{x}"],
                },
            },
            "priors": {"/x": "uniform(0, 1)"},
            "user": "userxyz",
            "user_args": ["python", script, "-x~uniform(0, 1)"],
            "user_script": script,
        },
        "refers": {"adapter": [], "parent_id": None, "root_id": 1},
    }

    local = {
        "experiment": {
            "name": "test-name",
            "user": "useruvt",
            "max_trials": 50,
            "max_broken": 15,
            "working_dir": "here_again",
            "algorithms": {"ac": {"d": "e", "f": "g"}},
        }
    }

    cmdargs = {
        "name": "exp-name",
        "user": "userabc",
        "version": 1,
        "exp-max-trials": 100,
        "exp-max-broken": 50,
        "working-dir": "cmdline-working-dir",
    }

    def sanity_check(self):
        """Check that defaults are different than testing configuration"""
        assert orion.core.config.to_dict()["experiment"] != self.config["experiment"]

    def _compare(self, base_config, experiment_config, ignore=tuple()):
        def _prune(config):
            config = copy.deepcopy(config)
            for key in ignore:
                config.pop(key, None)

            if "metadata" in config and "user" in config["metadata"]:
                config["user"] = config["metadata"]["user"]

            return config

        experiment_config = _prune(experiment_config)
        base_config = _prune(base_config)

        for key in list(experiment_config.keys()):
            if key not in base_config:
                experiment_config.pop(key)

        assert experiment_config == base_config

    def check_global_config(self, tmp_path, monkeypatch):
        """Check that global configuration is set properly"""
        self._compare(
            self.config["experiment"], orion.core.config.to_dict()["experiment"]
        )
        command = f"hunt --init-only -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        storage = setup_storage()

        experiment = get_experiment("test")
        self._compare(
            self.config["experiment"],
            experiment.configuration,
            ignore=["worker_trials"],
        )

    def check_env_var_config(self, tmp_path, monkeypatch):
        """Check that env vars overrides global configuration"""
        assert (
            orion.core.config.experiment.max_trials
            == self.env_vars["ORION_EXP_MAX_TRIALS"]
        )
        assert (
            orion.core.config.experiment.max_broken
            == self.env_vars["ORION_EXP_MAX_BROKEN"]
        )
        assert (
            orion.core.config.experiment.working_dir
            == self.env_vars["ORION_WORKING_DIR"]
        )

        command = f"hunt --init-only -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        experiment = get_experiment("test")

        assert experiment.max_trials == self.env_vars["ORION_EXP_MAX_TRIALS"]
        assert experiment.max_broken == self.env_vars["ORION_EXP_MAX_BROKEN"]
        assert experiment.working_dir == self.env_vars["ORION_WORKING_DIR"]

    def check_db_config(self):
        """Check that db config overrides global/envvar config"""
        name = "test-name"
        command = f"hunt --worker-max-trials 0 -n {name}"
        orion.core.cli.main(command.split(" "))

        storage = setup_storage()

        experiment = get_experiment(name)
        self._compare(self.database, experiment.configuration, ignore=["worker_trials"])

    def check_local_config(self, tmp_path, conf_file, monkeypatch):
        """Check that local configuration overrides global/envvars configuration"""
        command = f"hunt --worker-max-trials 0 -c {conf_file}"
        orion.core.cli.main(command.split(" "))

        storage = setup_storage()

        experiment = get_experiment("test-name")
        self._compare(self.local["experiment"], experiment.configuration)

    def check_cmd_args_config(self, tmp_path, conf_file, monkeypatch):
        """Check that cmdargs configuration overrides global/envvars/local configuration"""
        command = f"hunt --worker-max-trials 0 -c {conf_file} --branch-from test-name --enable-evc"
        command += " " + " ".join(
            f"--{name} {value}" for name, value in self.cmdargs.items()
        )
        orion.core.cli.main(command.split(" "))

        storage = setup_storage()

        experiment = get_experiment("exp-name")
        assert experiment.name == "exp-name"
        assert experiment.node.parent.name == "test-name"
        assert experiment.version == 1
        assert experiment.metadata["user"] == self.cmdargs["user"]
        assert experiment.max_trials == self.cmdargs["exp-max-trials"]
        assert experiment.max_broken == self.cmdargs["exp-max-broken"]
        assert experiment.working_dir == self.cmdargs["working-dir"]


class TestWorkerConfig(ConfigurationTestSuite):
    """Test suite for worker configuration"""

    config = {
        "worker": {
            "n_workers": 2,
            "pool_size": 2,
            "executor": "singleexecutor",
            "executor_configuration": {"threads_per_worker": 1},
            "heartbeat": 30,
            "max_trials": 10,
            "max_broken": 5,
            "reservation_timeout": 16,
            "idle_timeout": 17,
            "max_idle_time": 15,
            "interrupt_signal_code": 131,
            "user_script_config": "cfg",
        }
    }

    env_vars = {
        "ORION_N_WORKERS": 3,
        "ORION_POOL_SIZE": 1,
        "ORION_EXECUTOR": "joblib",
        "ORION_HEARTBEAT": 40,
        "ORION_WORKER_MAX_TRIALS": 20,
        "ORION_WORKER_MAX_BROKEN": 6,
        "ORION_RESERVATION_TIMEOUT": 17,
        "ORION_IDLE_TIMEOUT": 18,
        "ORION_MAX_IDLE_TIME": 16,
        "ORION_INTERRUPT_CODE": 132,
        "ORION_USER_SCRIPT_CONFIG": "envcfg",
    }

    local = {
        "worker": {
            "n_workers": 4,
            "pool_size": 5,
            "executor": "singleexecutor",
            "executor_configuration": {"threads_per_worker": 2},
            "heartbeat": 50,
            "max_trials": 30,
            "max_broken": 7,
            "reservation_timeout": 17,
            "idle_timeout": 18,
            "max_idle_time": 16,
            "interrupt_signal_code": 133,
            "user_script_config": "lclcfg",
        }
    }

    cmdargs = {
        "n-workers": 1,
        "pool-size": 6,
        "executor": "singleexecutor",
        "heartbeat": 70,
        "worker-max-trials": 0,
        "worker-max-broken": 8,
        "reservation-timeout": 18,
        "idle-timeout": 19,
        "max-idle-time": 17,
        "interrupt-signal-code": 134,
        "user-script-config": "cmdcfg",
    }

    def sanity_check(self):
        """Check that defaults are different than testing configuration"""
        assert orion.core.config.to_dict()["worker"] != self.config["worker"]

    def _mock(self, monkeypatch):
        self._mock_exp_client(monkeypatch)
        self._mock_consumer(monkeypatch)
        self._mock_producer(monkeypatch)
        self._mock_workon(monkeypatch)

    def _mock_exp_client(self, monkeypatch):
        self.exp_client = None
        old_init = orion.client.experiment.ExperimentClient.__init__

        def init(c_self, *args, **kwargs):
            old_init(c_self, *args, **kwargs)
            self.exp_client = c_self

        monkeypatch.setattr(orion.client.experiment.ExperimentClient, "__init__", init)

    def _mock_consumer(self, monkeypatch):
        self.consumer = None
        old_init = orion.core.cli.hunt.Consumer.__init__

        def init(c_self, *args, **kwargs):
            old_init(c_self, *args, **kwargs)
            self.consumer = c_self

        monkeypatch.setattr(orion.core.cli.hunt.Consumer, "__init__", init)

    def _mock_producer(self, monkeypatch):
        self.producer = None
        old_init = orion.core.worker.producer.Producer.__init__

        def init(p_self, *args, **kwargs):
            old_init(p_self, *args, **kwargs)
            self.producer = p_self

        monkeypatch.setattr(orion.core.worker.producer.Producer, "__init__", init)

    def _mock_workon(self, monkeypatch):
        workon = orion.core.cli.hunt.workon

        self.workon_kwargs = None

        def mocked_workon(experiment, **kwargs):
            self.workon_kwargs = kwargs
            return workon(experiment, **kwargs)

        monkeypatch.setattr("orion.core.cli.hunt.workon", mocked_workon)

    def _check_mocks(self, config):
        self._check_exp_client(config)
        self._check_consumer(config)
        self._check_workon(config)

    def _check_exp_client(self, config):
        assert self.exp_client.heartbeat == config["heartbeat"]

    def _check_consumer(self, config):
        assert (
            self.consumer.template_builder.config_prefix == config["user_script_config"]
        )
        assert self.consumer.interrupt_signal_code == config["interrupt_signal_code"]

    def _check_workon(self, config):
        assert self.workon_kwargs["n_workers"] == config["n_workers"]
        assert self.workon_kwargs["executor"] == config["executor"]
        assert (
            self.workon_kwargs["executor_configuration"]
            == config["executor_configuration"]
        )
        assert self.workon_kwargs["pool_size"] == config["pool_size"]
        assert (
            self.workon_kwargs["reservation_timeout"] == config["reservation_timeout"]
        )
        assert self.workon_kwargs["idle_timeout"] == config["idle_timeout"]
        assert self.workon_kwargs["max_trials"] == config["max_trials"]
        assert self.workon_kwargs["max_broken"] == config["max_broken"]

    def check_global_config(self, tmp_path, monkeypatch):
        """Check that global configuration is set properly"""
        assert orion.core.config.to_dict()["worker"] == self.config["worker"]

        self._mock(monkeypatch)

        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        self._check_mocks(self.config["worker"])

    def check_env_var_config(self, tmp_path, monkeypatch):
        """Check that env vars overrides global configuration"""
        env_var_config = {
            "n_workers": self.env_vars["ORION_N_WORKERS"],
            "pool_size": self.env_vars["ORION_POOL_SIZE"],
            "executor": self.env_vars["ORION_EXECUTOR"],
            "executor_configuration": self.config["worker"]["executor_configuration"],
            "heartbeat": self.env_vars["ORION_HEARTBEAT"],
            "max_trials": self.env_vars["ORION_WORKER_MAX_TRIALS"],
            "max_broken": self.env_vars["ORION_WORKER_MAX_BROKEN"],
            "reservation_timeout": self.env_vars["ORION_RESERVATION_TIMEOUT"],
            "idle_timeout": self.env_vars["ORION_IDLE_TIMEOUT"],
            "max_idle_time": self.env_vars["ORION_MAX_IDLE_TIME"],
            "interrupt_signal_code": self.env_vars["ORION_INTERRUPT_CODE"],
            "user_script_config": self.env_vars["ORION_USER_SCRIPT_CONFIG"],
        }

        assert orion.core.config.to_dict()["worker"] == env_var_config

        # Override executor configuration otherwise joblib will fail.
        orion.core.config.worker.executor_configuration = {}
        env_var_config["executor_configuration"] = {}

        self._mock(monkeypatch)

        command = f"hunt --exp-max-trials 0 -n test python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        self._check_mocks(env_var_config)

    def check_db_config(self):
        """No Storage config in DB, no test"""

    def check_local_config(self, tmp_path, conf_file, monkeypatch):
        """Check that local configuration overrides global/envvars configuration"""
        self._mock(monkeypatch)

        # Override executor so that executor and configuration are coherent in global config
        os.environ["ORION_EXECUTOR"] = "singleexecutor"

        command = f"hunt --exp-max-trials 0 -n test -c {conf_file} python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        self._check_mocks(self.local["worker"])

    def check_cmd_args_config(self, tmp_path, conf_file, monkeypatch):
        """Check that cmdargs configuration overrides global/envvars/local configuration"""
        config = {
            "n_workers": self.cmdargs["n-workers"],
            "executor": self.cmdargs["executor"],
            "executor_configuration": {"threads_per_worker": 2},
            "pool_size": self.cmdargs["pool-size"],
            "reservation_timeout": self.cmdargs["reservation-timeout"],
            "idle_timeout": self.cmdargs["idle-timeout"],
            "heartbeat": self.cmdargs["heartbeat"],
            "max_trials": self.cmdargs["worker-max-trials"],
            "max_broken": self.cmdargs["worker-max-broken"],
            "interrupt_signal_code": self.cmdargs["interrupt-signal-code"],
            "user_script_config": self.cmdargs["user-script-config"],
        }

        self._mock(monkeypatch)

        # Override executor so that executor and configuration are coherent in global config
        os.environ["ORION_EXECUTOR"] = "singleexecutor"

        command = f"hunt -c {conf_file} -n cmd-test"
        command += " " + " ".join(
            f"--{name} {value}" for name, value in self.cmdargs.items()
        )
        command += f" python {script} -x~uniform(0,1)"
        orion.core.cli.main(command.split(" "))

        self._check_mocks(config)


class TestEVCConfig(ConfigurationTestSuite):
    """Test for EVC configuration"""

    config = {
        "evc": {
            "enable": False,
            "auto_resolution": False,
            "manual_resolution": True,
            "non_monitored_arguments": ["test", "one"],
            "ignore_code_changes": True,
            "algorithm_change": True,
            "code_change_type": "noeffect",
            "cli_change_type": "noeffect",
            "config_change_type": "noeffect",
            "orion_version_change": True,
        }
    }

    env_vars = {
        "ORION_EVC_ENABLE": "true",
        "ORION_EVC_MANUAL_RESOLUTION": "",
        "ORION_EVC_NON_MONITORED_ARGUMENTS": "test:two:others",
        "ORION_EVC_IGNORE_CODE_CHANGES": "",
        "ORION_EVC_ALGO_CHANGE": "",
        "ORION_EVC_CODE_CHANGE": "unsure",
        "ORION_EVC_CMDLINE_CHANGE": "unsure",
        "ORION_EVC_CONFIG_CHANGE": "unsure",
        "ORION_EVC_ORION_VERSION_CHANGE": "",
    }

    local = {
        "evc": {
            "enable": False,
            "manual_resolution": True,
            "non_monitored_arguments": ["test", "local"],
            "ignore_code_changes": False,
            "algorithm_change": True,
            "code_change_type": "break",
            "cli_change_type": "break",
            "config_change_type": "noeffect",
            "orion_version_change": True,
        }
    }

    cmdargs = {
        "enable-evc": True,
        "manual-resolution": False,
        "non-monitored-arguments": "test:cmdargs",
        "ignore-code-changes": True,
        "algorithm-change": False,
        "code-change-type": "noeffect",
        "cli-change-type": "unsure",
        "config-change-type": "break",
        "orion-version-change": False,
    }

    def sanity_check(self):
        """Check that defaults are different than testing configuration"""
        assert orion.core.config.to_dict()["evc"] != self.config["evc"]

    def _mock_consumer(self, monkeypatch):
        self.consumer = None
        old_init = orion.core.cli.hunt.Consumer.__init__

        def init(c_self, *args, **kwargs):
            old_init(c_self, *args, **kwargs)
            self.consumer = c_self

        monkeypatch.setattr(orion.core.cli.hunt.Consumer, "__init__", init)

    def _check_consumer(self, config):
        assert self.consumer.ignore_code_changes == config["ignore_code_changes"]

    def check_global_config(self, tmp_path, monkeypatch):
        """Check that global configuration is set properly"""
        assert orion.core.config.to_dict()["evc"] == self.config["evc"]

        name = "global-test"
        command = f"hunt --enable-evc --worker-max-trials 0 -n {name} python {script} -x~uniform(0,1)"
        assert orion.core.cli.main(command.split(" ")) == 0

        # Test that manual_resolution is True and it branches when changing cli (thus crash)
        assert orion.core.cli.main(f"{command} --cli-change ".split(" ")) == 1

        command = "hunt --auto-resolution " + command[5:]

        self._check_enable(name, command.replace(" --enable-evc", ""), enabled=False)

        self._check_cli_change(name, command, change_type="noeffect")

        self._check_non_monitored_arguments(
            name, command, non_monitored_arguments=["test", "one"]
        )

        self._check_script_config_change(
            tmp_path, name, command, change_type="noeffect"
        )

        # EVC not enabled, code change should be ignored even if option is set to True
        assert self.config["evc"]["enable"] is False
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command.replace("--enable-evc ", ""),
                mock_ignore_code_changes=True,
                ignore_code_changes=True,
                change_type=self.config["evc"]["code_change_type"],
                enable_evc=False,
            )

        # EVC is enabled, option should be honored
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command,
                mock_ignore_code_changes=None,
                ignore_code_changes=self.config["evc"]["ignore_code_changes"],
                change_type=self.config["evc"]["code_change_type"],
                enable_evc=True,
            )

    def check_env_var_config(self, tmp_path, monkeypatch):
        """Check that env vars overrides global configuration"""
        assert orion.core.config.evc.enable
        assert not orion.core.config.evc.manual_resolution
        assert not orion.core.config.evc.ignore_code_changes
        assert not orion.core.config.evc.algorithm_change
        assert orion.core.config.evc.non_monitored_arguments == [
            "test",
            "two",
            "others",
        ]
        assert orion.core.config.evc.code_change_type == "unsure"
        assert orion.core.config.evc.cli_change_type == "unsure"
        assert orion.core.config.evc.config_change_type == "unsure"
        assert not orion.core.config.evc.orion_version_change

        name = "env-var-test"
        command = (
            f"hunt --worker-max-trials 0 -n {name} python {script} -x~uniform(0,1)"
        )
        assert orion.core.cli.main(command.split(" ")) == 0

        self._check_enable(name, command, enabled=True)

        self._check_cli_change(name, command, change_type="unsure")
        self._check_non_monitored_arguments(
            name, command, non_monitored_arguments=["test", "two", "others"]
        )
        self._check_script_config_change(tmp_path, name, command, change_type="unsure")

        # Enable EVC, ignore_code_changes is False
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command,
                mock_ignore_code_changes=None,
                ignore_code_changes=bool(
                    self.env_vars["ORION_EVC_IGNORE_CODE_CHANGES"]
                ),
                change_type=self.env_vars["ORION_EVC_CODE_CHANGE"],
                enable_evc=True,
            )

        # Disable EVC, ignore_code_changes is True for Consumer
        os.environ["ORION_EVC_ENABLE"] = ""
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command,
                mock_ignore_code_changes=None,
                ignore_code_changes=bool(
                    self.env_vars["ORION_EVC_IGNORE_CODE_CHANGES"]
                ),
                change_type=self.env_vars["ORION_EVC_CODE_CHANGE"],
                enable_evc=False,
            )

    def check_db_config(self):
        """No Storage config in DB, no test"""

    def check_local_config(self, tmp_path, conf_file, monkeypatch):
        """Check that local configuration overrides global/envvars configuration"""
        name = "local-test"
        command = (
            f"hunt --enable-evc --worker-max-trials 0 -n {name} -c {conf_file} "
            f"python {script} -x~uniform(0,1)"
        )

        assert orion.core.cli.main(command.split(" ")) == 0

        # Test that manual_resolution is True and it branches when changing cli
        assert orion.core.cli.main(f"{command} --cli-change ".split(" ")) == 1

        command = "hunt --auto-resolution " + command[5:]

        self._check_enable(name, command.replace(" --enable-evc", ""), enabled=False)

        self._check_cli_change(
            name, command, change_type=self.local["evc"]["cli_change_type"]
        )
        self._check_non_monitored_arguments(
            name,
            command,
            non_monitored_arguments=self.local["evc"]["non_monitored_arguments"],
        )
        self._check_script_config_change(
            tmp_path,
            name,
            command,
            change_type=self.local["evc"]["config_change_type"],
        )

        # enabled evc, ignore code changes so to True
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command,
                mock_ignore_code_changes=False,
                ignore_code_changes=self.local["evc"]["ignore_code_changes"],
                change_type=self.local["evc"]["code_change_type"],
                enable_evc=True,
            )

        # disabled evc, ignore code changes so to True
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command.replace("--enable-evc ", ""),
                mock_ignore_code_changes=False,
                ignore_code_changes=self.local["evc"]["ignore_code_changes"],
                change_type=self.local["evc"]["code_change_type"],
                enable_evc=False,
            )

    def check_cmd_args_config(self, tmp_path, conf_file, monkeypatch):
        """Check that cmdargs configuration overrides global/envvars/local configuration"""
        name = "cmd-test"
        command = (
            f"hunt --worker-max-trials 0 -c {conf_file} -n {name} "
            "--enable-evc "
            "--auto-resolution "
            f"python {script} -x~uniform(0,1)"
        )
        assert orion.core.cli.main(command.split(" ")) == 0

        self._check_enable(name, command, enabled=True)

        self._check_cli_change(
            name,
            command="hunt --cli-change-type unsure " + command[5:],
            change_type=self.cmdargs["cli-change-type"],
        )
        self._check_non_monitored_arguments(
            name,
            command="hunt --non-monitored-arguments test:cmdargs " + command[5:],
            non_monitored_arguments=self.cmdargs["non-monitored-arguments"].split(":"),
        )

        self._check_script_config_change(
            tmp_path,
            name,
            command="hunt --config-change-type break " + command[5:],
            change_type=self.cmdargs["config-change-type"],
        )

        # Mock local to ignore_code_changes=False
        fetch_config = orion.core.io.resolve_config.fetch_config

        def mock_local(cmdargs):
            config = fetch_config(cmdargs)
            config["evc"]["ignore_code_changes"] = False
            return config

        monkeypatch.setattr(orion.core.io.resolve_config, "fetch_config", mock_local)

        # Check that ignore_code_changes is rightly False
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command="hunt --code-change-type noeffect " + command[5:],
                mock_ignore_code_changes=False,
                ignore_code_changes=False,
                change_type=self.cmdargs["code-change-type"],
                enable_evc=True,
            )

        # Check that ignore_code_changes is now True because --ignore-code-changes was added
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command="hunt --ignore-code-changes --code-change-type noeffect "
                + command[5:],
                mock_ignore_code_changes=True,
                ignore_code_changes=True,
                change_type=self.cmdargs["code-change-type"],
                enable_evc=True,
            )

        # Check that ignore_code_changes is forced to True in consumer
        # even if --ignore-code-changes is not passed
        with monkeypatch.context() as m:
            self._check_code_change(
                m,
                name,
                command.replace("--enable-evc ", ""),
                mock_ignore_code_changes=False,
                ignore_code_changes=False,
                change_type=self.cmdargs["code-change-type"],
                enable_evc=False,
            )

    @with_storage_fork
    def _check_enable(self, name, command, enabled):
        command += " --cli-change "
        experiment = get_experiment(name)
        if enabled:
            assert orion.core.cli.main(command.split(" ")) == 0
            assert get_experiment(name).version == experiment.version + 1
        else:
            assert orion.core.cli.main(command.split(" ")) == 0
            assert get_experiment(name).version == experiment.version

    @with_storage_fork
    def _check_cli_change(self, name, command, change_type):
        command += " --cli-change "

        experiment = get_experiment(name)
        # Test that manual_resolution is False and it branches when changing cli
        assert orion.core.cli.main(command.split(" ")) == 0

        new_experiment = get_experiment(name)

        assert new_experiment.version == experiment.version + 1
        assert new_experiment.refers["adapter"].configuration[0] == {
            "of_type": "commandlinechange",
            "change_type": change_type,
        }

    @with_storage_fork
    def _check_non_monitored_arguments(self, name, command, non_monitored_arguments):
        for argument in non_monitored_arguments:
            command += f" --{argument} "

        experiment = get_experiment(name)
        # Test that cli change with non-monitored args do not cause branching
        assert orion.core.cli.main(command.split(" ")) == 0

        assert get_experiment(name).version == experiment.version

    @with_storage_fork
    def _check_code_change(
        self,
        monkeypatch,
        name,
        command,
        mock_ignore_code_changes,
        ignore_code_changes,
        change_type,
        enable_evc,
    ):
        """Check if code changes are correctly ignored during experiment build and by consumer
        between two trial executions.
        """

        # Test that code change is handled with 'no-effect'
        def fixed_dictionary(user_script):
            """Create VCS"""
            vcs = {}
            vcs["type"] = "git"
            vcs["is_dirty"] = False
            vcs["HEAD_sha"] = "test " + str(random.random())
            vcs["active_branch"] = None
            vcs["diff_sha"] = "diff"
            return vcs

        monkeypatch.setattr(
            orion.core.io.resolve_config, "infer_versioning_metadata", fixed_dictionary
        )

        self._mock_consumer(monkeypatch)

        detect = orion.core.evc.conflicts.CodeConflict.detect

        def mock_detect(old_config, new_config, branching_config=None):
            if branching_config and "ignore_code_changes" in branching_config:
                assert (
                    branching_config["ignore_code_changes"] is mock_ignore_code_changes
                )
            # branching_config["ignore_code_changes"] = False
            return detect(old_config, new_config, branching_config)

        monkeypatch.setattr(
            orion.core.evc.conflicts.CodeConflict, "detect", mock_detect
        )

        experiment = get_experiment(name)

        assert orion.core.cli.main(command.split(" ")) == 0
        self._check_consumer(
            {
                "ignore_code_changes": (
                    (enable_evc and ignore_code_changes) or not enable_evc
                )
            }
        )

        new_experiment = get_experiment(name)
        if enable_evc and not ignore_code_changes:
            assert new_experiment.version == experiment.version + 1
            assert new_experiment.refers["adapter"].configuration[0] == {
                "of_type": "codechange",
                "change_type": change_type,
            }
        elif enable_evc:  # But code change ignored, so no branching event.
            assert new_experiment.version == experiment.version
        else:
            assert new_experiment.version == experiment.version

    @with_storage_fork
    def _check_script_config_change(self, tmp_path, name, command, change_type):

        experiment = get_experiment(name)

        # Test that config change is handled with 'break'
        with self.setup_user_script_config(tmp_path) as user_script_config:

            command += f" --config {user_script_config}"
            assert orion.core.cli.main(command.split(" ")) == 0

        new_experiment = get_experiment(name)

        assert new_experiment.version == experiment.version + 1
        assert len(new_experiment.refers["adapter"].configuration) == 2
        assert new_experiment.refers["adapter"].configuration[1] == {
            "of_type": "scriptconfigchange",
            "change_type": change_type,
        }

    @contextmanager
    def setup_user_script_config(self, tmp_path):
        """Setup temporary dummy user script config"""
        conf_file = tmp_path / "user_script_config.yaml"
        config = {"what": "ever"}
        conf_file.write_text(yaml.dump(config))
        yield conf_file
