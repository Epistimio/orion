"""Tests the minimalist example script on scitkit-learn and its integration to Oríon."""
import os
import subprocess

import pytest

from orion.client import create_experiment
import orion.core.cli
from orion.storage.base import get_storage

@pytest.fixture(autouse=True)
def cleanup_files(monkeypatch):
    yield
    file_path = '~/.config/orion.core/test-db.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)


def test_script_integrity(capsys):
    """Verifies the example script can run in standalone via `python ...`."""
    script = os.path.abspath("examples/scikitlearn-iris/main.py")

    return_code = subprocess.call(["python", script, '0.1'])

    assert return_code != 2, "The example script does not exists."
    assert return_code != 1, "The example script did not terminates its execution."
    assert return_code == 0 and not capsys.readouterr().err, \
        "The example script encountered an error during its execution."


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_orion_runs_script(monkeypatch):
    """Verifies Oríon can execute the example script."""
    script = os.path.abspath("examples/scikitlearn-iris/main.py")
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = "orion_config.yaml"

    orion.core.cli.main(["hunt", "--config", config, "python", script,
                         "orion~choices([0.1])"])

    experiment = create_experiment(name="scikit-iris-tutorial")
    assert experiment is not None
    assert experiment.version == 1
    assert len(experiment.space.items()) == 1
    assert experiment.space.items()[0][0] == '/_pos_2'

    storage = get_storage()
    trials = storage.fetch_trials(uid=experiment.id)
    assert len(trials) == 1

    trial = trials[0]
    assert trial.status == 'completed'
    assert trial.params['/_pos_2'] == 0.1


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_result_reproducibility(monkeypatch):
    """Verifies the script results stays consistent (with respect to the documentation)."""
    script = os.path.abspath("examples/scikitlearn-iris/main.py")
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = "orion_config.yaml"

    orion.core.cli.main(["hunt", "--config", config, "python", script,
                         "orion~choices([0.1])"])

    experiment = create_experiment(name="scikit-iris-tutorial")
    assert 'best_evaluation' in experiment.stats
    assert experiment.stats['best_evaluation'] == 0.6666666666666667
