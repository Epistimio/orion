from threading import Thread

from orion.service.client import ExperimentClientREST
from orion.service.testing import server


def function(a, b):
    return [dict(value=a + b, type="objective", name="wahtever")]


TOKEN = "Tok1"


def test_one_rest_client():
    with server() as (endpoint, port):
        client = ExperimentClientREST.create_experiment(
            "MyExperiment",
            version=None,
            space=dict(a="uniform(0, 1)", b="uniform(0, 1)"),
            algorithms=None,
            strategy=None,
            max_trials=10,
            max_broken=None,
            storage=dict(
                type="reststorage",
                endpoint=endpoint,
                token=TOKEN,
            ),
            branching=None,
            max_idle_time=None,
            heartbeat=None,
            working_dir=None,
            debug=False,
        )

        count = client.workon(
            fct=function,
            n_workers=2,
        )

        assert count > 10
        assert client.is_done


def test_n_client(n=2):
    return

    with server() as endpoint:
        workers = []

        for i in range(n):
            client = ExperimentClientREST.create_experiment()
            worker = Thread(target=client.workon)
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
