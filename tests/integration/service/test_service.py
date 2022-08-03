"""Integration testing between the REST service and the client"""

import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager

from orion.core.io.database.mongodb import MongoDB
from orion.service.broker.broker import ServiceContext
from orion.service.client import ClientREST
from orion.storage.legacy import Legacy
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


MONGO_DB_PORT = 8124
MONGO_DB_ADDRESS = "localhost"


@contextmanager
def server():
    servicectx = ServiceContext()
    servicectx.host = MONGO_DB_ADDRESS
    servicectx.port = MONGO_DB_PORT

    with mongod(servicectx.port, servicectx.host):
        with service(8080, "localhost", servicectx):
            yield


def test_setup():
    with mongod(MONGO_DB_PORT, MONGO_DB_ADDRESS) as dbpath:
        print("Starting service")
        # add_mongo_user(dbpath, "abc", '123')


def get_mongo_admin():
    db = MongoDB(
        name="orion",
        host=MONGO_DB_ADDRESS,
        port=MONGO_DB_PORT,
        username="god",
        password="god123",
        owner=None,
    )

    return Legacy(database_instance=db, setup=False)


def test_new_experiment():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)
        expid1 = client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        # same experiment should be no problem
        client2 = ClientREST(ENDPOINT, TOKEN2)
        expid2 = client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", c="uniform(0, 1)")
        )
        assert expid1 == expid2

        storage = get_mongo_admin()
        experiences = storage.fetch_experiments(dict(name="MyExperiment"))
        assert len(experiences) == 2, "Each user has their own experiment"


def test_suggest():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        trials = client.suggest()
        assert len(trials) > 0


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
