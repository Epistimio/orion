"""
Helper script to add an uncompleted experiment to
db_dashboard_full_with_uncompleted_experiments.pkl
built from a copy of db_dashboard_full.pkl
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


NEW_EXPERIMENT_DEFINITIONS = [
    # Default, max trial 200
    {
        "name": "uncompleted_experiment",
        "count": {
            "completed": 40,
            "new": 30,
            "reserved": 25,
            "suspended": 20,
            "interrupted": 15,
            "broken": 10,
        },
        "max_trials": 200,
    },
    # Max trial 0
    {
        "name": "uncompleted_max_trials_0",
        "count": {
            "completed": 6,
            "new": 5,
            "reserved": 4,
            "suspended": 3,
            "interrupted": 2,
            "broken": 1,
        },
        "max_trials": 0,
    },
    # Max trial infinite
    {
        "name": "uncompleted_max_trials_infinite",
        "count": {
            "completed": 6,
            "new": 5,
            "reserved": 4,
            "suspended": 3,
            "interrupted": 2,
            "broken": 1,
        },
        "max_trials": float("inf"),
    },
    # Completed > max trials
    {
        "name": "uncompleted_max_trials_lt_completed_trials",
        "count": {
            "completed": 20,
            "new": 5,
            "reserved": 4,
            "suspended": 3,
            "interrupted": 2,
            "broken": 1,
        },
        "max_trials": 10,
    },
    # No completed trials
    {
        "name": "uncompleted_no_completed_trials",
        "count": {
            "completed": 0,
            "new": 5,
            "reserved": 4,
            "suspended": 3,
            "interrupted": 2,
            "broken": 1,
        },
        "max_trials": 200,
    },
]


def main():
    # Get base experiment model from experiment 2-dim-shape-exp.1 read from db_dashboard_full.pkl
    input_storage = setup_storage(
        {
            "database": {
                "host": ".github/workflows/orion/db_dashboard_full.pkl",
                "type": "pickleddb",
            }
        }
    )
    (base_experiment,) = input_storage._db.read(
        "experiments", {"name": "2-dim-shape-exp", "version": 1}
    )
    del base_experiment["_id"]
    base_experiment["metadata"]["datetime"] = SUBMIT_TIME

    storage = setup_storage(
        {
            "database": {
                "host": ".github/workflows/orion/db_dashboard_full_with_uncompleted_experiments.pkl",
                "type": "pickleddb",
            }
        }
    )
    pickle_db = storage._db

    # Insert new experiments into db_dashboard_full_with_uncompleted_experiments.pkl
    for new_exp_def in NEW_EXPERIMENT_DEFINITIONS:
        new_experiment = base_experiment.copy()
        new_experiment["name"] = new_exp_def["name"]
        new_experiment["max_trials"] = new_exp_def["max_trials"]

        # Clean database if necessary.
        for exp_found in pickle_db.read(
            "experiments", {"name": new_experiment["name"]}
        ):
            nb_exps_deleted = pickle_db.remove("experiments", {"_id": exp_found["_id"]})
            nb_trials_deleted = pickle_db.remove(
                "trials", {"experiment": exp_found["_id"]}
            )
            print(
                f"[{new_experiment['name']}] Deleted",
                nb_exps_deleted,
                "experiment(s),",
                nb_trials_deleted,
                "trial(s)",
            )

        # Write and get new experiment.
        pickle_db.write(
            "experiments",
            new_experiment,
        )
        (exp,) = pickle_db.read("experiments", {"name": new_experiment["name"]})

        # Write new experiment trials.
        x = {"name": "/x", "type": "real", "value": 0.0}
        results = {"name": "obj", "type": "objective", "value": 0.0}
        for status, count in sorted(new_exp_def["count"].items()):
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
        print(f"[{new_experiment['name']}] written.")


if __name__ == "__main__":
    main()
