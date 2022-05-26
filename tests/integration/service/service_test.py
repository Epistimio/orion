"""Integration testing between the REST service and the client"""

import multiprocessing
import os
import signal
from contextlib import contextmanager

from orion.service.client import ClientREST

TOKEN = ""
ENDPOINT = "http://localhost:8080"


@contextmanager
def server() -> None:
    import time

    from orion.service.service import main

    p = multiprocessing.Process(target=main)
    p.start()

    # The server takes a bit of time to setup
    time.sleep(1)

    try:
        yield p
    finally:
        os.kill(p.pid, signal.SIGINT)
        p.terminate()


def test_new_experiment():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)
        expid = client.new_experiment(abc=123)
        print(expid)


def test_suggest():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)
        trials = client.suggest()

        assert len(trials) > 0
        print(trials)


def test_observe():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        trials = client.suggest()

        assert len(trials) > 0
        client.observe(trials[0], [dict(name="objective", type="objective", value=1)])


def test_heartbeat():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        trials = client.suggest()

        assert len(trials) > 0
        client.heartbeat(trials[0])


def test_is_done():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)
        expid = client.is_done()
        print(expid)
