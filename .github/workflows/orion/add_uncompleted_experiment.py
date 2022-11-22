"""
Helper script to add an uncompleted experiment to
db_dashboard_full_with_uncompleted_experiment.pkl
"""
from datetime import datetime, timedelta

from orion.core.worker.trial import Trial
from orion.storage.base import setup_storage

SUBMIT_TIME = datetime(
    year=2000,
    month=1,
    day=1,
    hour=10,
    minute=0,
    second=0,
    microsecond=123456,
)

# Copied form experiment 2-dim-exp, name changed to uncompleted_experiment
UNCOMPLETED_EXPERIMENT = {
    "space": {
        "/dropout": "uniform(0, 0.5, precision=1)",
        "/learning_rate": "loguniform(1e-5, 1e-3, shape=3)",
    },
    "name": "uncompleted_experiment",
    "refers": {"parent_id": None, "root_id": 1, "adapter": []},
    "metadata": {
        "orion_version": "0.1.7",
        "user_script": "/home/hrb/dev/lateral-view-analysis/hyperparam_search.py",
        "VCS": {
            "type": "git",
            "is_dirty": True,
            "HEAD_sha": "2c3d64ee178d01b6f10f91e9ec125e5389776d08",
            "active_branch": "master",
            "diff_sha": "e2c9418e4267512a5227f8370b13872aff0b1e5669fe018782344d0744c42c55",
        },
        "user_args": [
            "--data_dir",
            "CLUSTER",
            "--csv_path",
            "/home/hrb/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/joint_PA_L.csv",
            "--splits_path",
            "/home/hrb/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_666.pkl",
            "--output_dir",
            "/lustre04/scratch/cohenjos/PC-output/hadrien",
            "--exp_name",
            "{trial.id}",
            "--seed",
            "666",
            "--epochs",
            "40",
            "--model-type",
            "dualnet",
            "--target",
            "joint",
            "--batch_size",
            "8",
            "--learning_rate",
            "orion~loguniform(1e-5, 1e-3, shape=3)",
            "--dropout",
            "orion~uniform(0, 0.5, precision=1)",
            "--optim",
            "adam",
            "--log",
            "{exp.working_dir}/{exp.name}_{trial.id}/exp.log",
        ],
        "user": "hrb",
        "parser": {
            "converter": None,
            "file_config_path": None,
            "config_prefix": "config",
            "config_file_data": {},
            "file_priors": [],
            "cmd_priors": [
                ["/learning_rate", "loguniform(1e-5, 1e-3, shape=3)"],
                ["/dropout", "uniform(0, 0.5, precision=1)"],
            ],
            "parser": {
                "template": [
                    "{_pos_0}",
                    "--data_dir",
                    "{data_dir}",
                    "--csv_path",
                    "{csv_path}",
                    "--splits_path",
                    "{splits_path}",
                    "--output_dir",
                    "{output_dir}",
                    "--exp_name",
                    "{exp_name}",
                    "--seed",
                    "{seed}",
                    "--epochs",
                    "{epochs}",
                    "--model-type",
                    "{model-type}",
                    "--target",
                    "{target}",
                    "--batch_size",
                    "{batch_size}",
                    "--learning_rate",
                    "{learning_rate}",
                    "--dropout",
                    "{dropout}",
                    "--optim",
                    "{optim}",
                    "--log",
                    "{log}",
                ],
                "keys": [
                    ["_pos_0", "_pos_0"],
                    ["data_dir", "--data_dir"],
                    ["csv_path", "--csv_path"],
                    ["splits_path", "--splits_path"],
                    ["output_dir", "--output_dir"],
                    ["exp_name", "--exp_name"],
                    ["seed", "--seed"],
                    ["epochs", "--epochs"],
                    ["model-type", "--model-type"],
                    ["target", "--target"],
                    ["batch_size", "--batch_size"],
                    ["learning_rate", "--learning_rate"],
                    ["dropout", "--dropout"],
                    ["optim", "--optim"],
                    ["log", "--log"],
                ],
                "arguments": [
                    [
                        "_pos_0",
                        "/home/hrb/dev/lateral-view-analysis/hyperparam_search.py",
                    ],
                    ["data_dir", "CLUSTER"],
                    [
                        "csv_path",
                        "/home/hrb/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/joint_PA_L.csv",
                    ],
                    [
                        "splits_path",
                        "/home/hrb/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_666.pkl",
                    ],
                    [
                        "output_dir",
                        "/lustre04/scratch/cohenjos/PC-output/hadrien",
                    ],
                    ["exp_name", "{trial.id}"],
                    ["seed", "666"],
                    ["epochs", "40"],
                    ["model-type", "dualnet"],
                    ["target", "joint"],
                    ["batch_size", "8"],
                    ["learning_rate", "orion~loguniform(1e-5, 1e-3, shape=3)"],
                    ["dropout", "orion~uniform(0, 0.5, precision=1)"],
                    ["optim", "adam"],
                    ["log", "{exp.working_dir}/{exp.name}_{trial.id}/exp.log"],
                ],
            },
        },
        "priors": {
            "/dropout": "uniform(0, 0.5, precision=1)",
            "/learning_rate": "loguniform(1e-5, 1e-3, shape=3)",
        },
        "datetime": SUBMIT_TIME,
    },
    "pool_size": 1,
    "max_trials": 20,
    "version": 1,
    "algorithms": {"random": {"seed": None}},
    "producer": {"strategy": "MaxParallelStrategy"},
    "working_dir": "orion_working_dir",
}
NB_TRIALS_TO_ADD = {
    "completed": 40,
    "new": 30,
    "reserved": 25,
    "suspended": 20,
    "interrupted": 15,
    "broken": 10,
}


def main():
    storage = setup_storage(
        {
            "database": {
                "host": ".github/workflows/orion/db_dashboard_full_with_uncompleted_experiment.pkl",
                "type": "pickleddb",
            }
        }
    )

    pickle_db = storage._db

    # Clean database if necessary.
    for exp_found in pickle_db.read("experiments", {"name": "uncompleted_experiment"}):
        nb_exps_deleted = pickle_db.remove("experiments", {"_id": exp_found["_id"]})
        nb_trials_deleted = pickle_db.remove("trials", {"experiment": exp_found["_id"]})
        print(
            "Deleted",
            nb_exps_deleted,
            "experiment(s),",
            nb_trials_deleted,
            "trial(s)",
        )

    pickle_db.write(
        "experiments",
        UNCOMPLETED_EXPERIMENT,
    )
    (exp,) = pickle_db.read("experiments", {"name": "uncompleted_experiment"})

    x = {"name": "/x", "type": "real", "value": 0.0}
    results = {"name": "obj", "type": "objective", "value": 0.0}
    for status, count in sorted(NB_TRIALS_TO_ADD.items()):
        for i in range(count):
            trial_kwargs = dict(
                experiment=exp["_id"],
                params=[x],
                status=status,
                results=[results],
                submit_time=SUBMIT_TIME,
            )
            if status != "new":
                trial_kwargs.update(
                    start_time=SUBMIT_TIME + timedelta(minutes=i),
                )
            if status == "completed":
                trial_kwargs.update(
                    end_time=SUBMIT_TIME + timedelta(minutes=(i + 2)),
                )
            pickle_db.write("trials", Trial(**trial_kwargs).to_dict())
            x["value"] += 1
            results["value"] += 0.1
    print("Written uncompleted experiment.")


if __name__ == "__main__":
    main()
