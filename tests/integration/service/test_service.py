"""Integration testing between the REST service and the client"""

import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager

from orion.service.broker.broker import ServiceContext
from orion.service.client import ClientREST
from orion.testing.mongod import mongod

TOKEN = "Tok1"
TOKEN2 = "Tok2"
ENDPOINT = "http://localhost:8080"

log = logging.getLogger(__file__)


def wait(p):
    acc = 0
    while p.is_alive() and acc < 2:
        acc += 0.01
        time.sleep(0.01)


@contextmanager
def service(port, address, servicectx) -> None:
    import time

    from orion.service.service import main

    p = multiprocessing.Process(target=main, args=(address, port, servicectx))
    p.start()

    # The server takes a bit of time to setup
    time.sleep(1)

    try:
        yield p
    finally:
        # raise KeyboardInterrupt for regular shutdown
        os.kill(p.pid, signal.SIGINT)
        wait(p)

        if p.is_alive():
            log.debug("process still alive after sigint")
            # notify the process we want to terminate it with SIGTERM
            p.terminate()
            wait(p)

        if p.is_alive():
            log.debug("process still alive after sigterm")
            # process is taking too long kill it
            p.kill()
            wait(p)

        p.join()


@contextmanager
def server():
    servicectx = ServiceContext()
    servicectx.host = "losthost"
    servicectx.port = 8124

    with mongod(servicectx.port, servicectx.host):
        with service(8080, "localhost", servicectx):
            yield


def test_setup():
    import logging

    logging.basicConfig()

    print("here")
    with mongod(8124, "localhost") as dbpath:
        print("Starting service")
        # add_mongo_user(dbpath, "abc", '123')
    print("done")


def test_new_experiment():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)
        expid1 = client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        # same experiment should be no problem
        client2 = ClientREST(ENDPOINT, TOKEN2)
        expid2 = client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )
        assert expid1 != expid2


def test_suggest():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )
        print("----Done")

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
