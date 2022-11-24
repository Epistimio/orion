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


NB_TRIALS_TO_ADD = {
    "completed": 40,
    "new": 30,
    "reserved": 25,
    "suspended": 20,
    "interrupted": 15,
    "broken": 10,
}


def main():
    # Generate experiment UNCOMPLETED_EXPERIMENT from experiment 2-dim-shape-exp.1 read from db_dashboard_full.pkl
    input_storage = setup_storage(
        {
            "database": {
                "host": ".github/workflows/orion/db_dashboard_full.pkl",
                "type": "pickleddb",
            }
        }
    )
    uncompleted_experiment, = input_storage._db.read("experiments", {"name": "2-dim-shape-exp", "version": 1})
    del uncompleted_experiment["_id"]
    uncompleted_experiment["name"] = "uncompleted_experiment"
    uncompleted_experiment["metadata"]["datetime"] = SUBMIT_TIME

    # Insert uncompleted experiment into db_dashboard_full_with_uncompleted_experiment.pkl

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
        uncompleted_experiment,
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
