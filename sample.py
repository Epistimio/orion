from orion.client import create_experiment
from orion.core.worker.trial import Trial


def main():
    experiment = create_experiment(
        name="foo",
        space={"x": "uniform(0,1)"},
        storage={"type": "legacy", "database": {"type": "pickleddb", "host": "db.pkl"}},
        max_trials=10,
    )

    def foo(x: float):
        return [dict(name="loss", type="objective", value=x**2)]

    experiment.workon(foo)

    print(
        min(
            experiment.fetch_trials_by_status("completed"),
            key=lambda trial: trial.objective.value,
        )
    )


if __name__ == "__main__":
    main()
