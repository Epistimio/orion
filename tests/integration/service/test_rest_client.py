import time
from multiprocessing import Process

from orion.service.client.experiment import ExperimentClientREST
from orion.service.testing import server


def function(a, b):
    return [dict(value=a + b, type="objective", name="whatever")]


TOKEN1 = "Tok1"
TOKEN2 = "Tok2"


def new_client(endpoint, tok):
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
            token=tok,
        ),
        branching=None,
        max_idle_time=None,
        heartbeat=None,
        working_dir="orion_test_working_dir",
        debug=False,
    )
    return client


def workon(endpoint, tok):
    client = new_client(endpoint, tok)

    count = client.workon(
        fct=function,
        n_workers=2,
    )

    client.close()

    assert count >= 10
    assert client.is_done


def test_n_workon_rest_client(tokens=None):
    if tokens is None:
        tokens = [TOKEN1, TOKEN2]

    with server() as (endpoint, _):
        time.sleep(10)

        workers = []

        for token in tokens:
            p = Process(target=workon, args=(endpoint, token))
            p.start()
            workers.append(p)

        for worker in workers:
            worker.join()
            assert worker.exitcode == 0
