"""Tests the minimalist example script on scitkit-learn and its integration to Oríon."""
import json
import os
import subprocess

import pytest

import orion.core.cli
from orion.client import create_experiment
from orion.storage.base import setup_storage


def json_clean(path, scale):
    """Modifies json file to reduce it to some scale."""
    f = open(path)
    json_list = json.load(f)
    new_list = {}
    ctr = 0

    for item in json_list:
        if ctr < len(json_list) * scale:
            new_list[item] = json_list[item]
        ctr += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_list, f, ensure_ascii=False, indent=2)


@pytest.fixture(scope="module")
def download_data(tmp_path_factory):
    # Creating paths
    path = tmp_path_factory.mktemp("out")
    data = path / "data"
    output = data / "results"

    """Calling the script, for downloading the data"""
    script = os.path.abspath("examples/speechbrain_tutorial/download_data.py")
    # Utiliser les commandes et overrides pour les download path
    dir_var = [
        "--data_folder",
        data,
        "--output_folder",
        output,
        "--train_annotation",
        path,
        "--valid_annotation",
        path,
        "--test_annotation",
        path,
    ]
    return_code = subprocess.call(
        [
            "python",
            script,
            "examples/speechbrain_tutorial/train.yaml",
            "--device",
            "cpu",
            "--data_folder",
            data,
            "--output_folder",
            output,
            "--train_annotation",
            path / "train.json",
            "--valid_annotation",
            path / "valid.json",
            "--test_annotation",
            path / "test.json",
        ]
    )

    """ Reducing the size of the training, testing and validation set for the purpose of this test. """

    json_clean(path / "test.json", 0.005)
    json_clean(path / "train.json", 0.005)
    json_clean(path / "valid.json", 0.005)

    assert return_code != 2, "The example script does not exists."
    assert return_code != 1, "The example script did not terminates its execution."

    """ Verifying if the temp dict is populated """
    assert len(os.listdir(data)) != 0, "The data was not downloaded correctly"

    return path


def test_script_integrity(capsys, download_data):
    """Verifies the example script can run in standalone via `python ...`."""
    print(download_data)
    script = os.path.abspath("examples/speechbrain_tutorial/main.py")
    path = download_data
    data = path / "data"
    output = data / "results"

    return_code = subprocess.call(
        [
            "python",
            script,
            "examples/speechbrain_tutorial/train.yaml",
            "--device",
            "cpu",
            "--number_of_epochs",
            "1",
            "--data_folder",
            data,
            "--output_folder",
            output,
            "--train_annotation",
            path / "train.json",
            "--valid_annotation",
            path / "valid.json",
            "--test_annotation",
            path / "test.json",
        ]
    )
    assert return_code != 2, "The example script does not exists."
    assert return_code != 1, "The example script did not terminates its execution."
    assert (
        return_code == 0 and not capsys.readouterr().err
    ), "The example script encountered an error during its execution."


@pytest.mark.usefixtures("orionstate")
def test_orion_runs_script(download_data):
    """Verifies Oríon can execute the example script."""
    script = os.path.abspath("examples/speechbrain_tutorial/main.py")
    path = download_data
    data = path / "data"
    output = data / "results"

    config = "tests/functional/example/orion_config_speechbrain.yaml"

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            config,
            "python",
            script,
            "examples/speechbrain_tutorial/train.yaml",
            "--device",
            "cpu",
            "--number_of_epochs",
            "1",
            "--data_folder",
            str(data),
            "--output_folder",
            str(output),
            "--train_annotation",
            str(path / "train.json"),
            "--valid_annotation",
            str(path / "valid.json"),
            "--test_annotation",
            str(path / "test.json"),
            "--lr~loguniform(0.05, 0.2)",
        ]
    )

    experiment = create_experiment(name="speechbrain-tutorial1")
    assert experiment is not None
    assert experiment.version == 1

    keys = experiment.space.keys()
    assert len(keys) == 1
    assert "/lr" in keys

    storage = setup_storage()
    trials = storage.fetch_trials(uid=experiment.id)
    assert len(trials) == 1

    trial = trials[0]
    assert trial.status == "completed"
    assert trial.params["/lr"] == 0.07452
