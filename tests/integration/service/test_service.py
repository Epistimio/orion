"""Integration testing between the REST service and the client"""

import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager

import pytest

from orion.core.io.database.mongodb import MongoDB
from orion.service.broker.broker import ServiceContext
from orion.service.client import ClientREST, ExperiementIsNotSetup
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
        expid2 = client2.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", c="uniform(0, 1)")
        )
        assert expid1.name == expid2.name
        assert expid1.euid != expid2.euid

        storage = get_mongo_admin()
        experiences = storage.fetch_experiments(dict(name="MyExperiment"))
        assert len(experiences) == 2, "Each user has their own experiment"

        # mongo = storage._db._db
        # experiments = mongo['experiments'].find()

        # print()
        # for e in experiences:
        #     print(e)

        # print()
        # for e in experiments:
        #     print(e)


def test_suggest():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        # no experiment
        with pytest.raises(ExperiementIsNotSetup):
            trials = client.suggest()

        client.new_experiment(
            name="MyExperiment",
            space=dict(a="uniform(0, 1)", b="uniform(0, 1)"),
        )

        client_trials = client.suggest()
        assert len(client_trials) > 0, "A trial was generated"

        storage = get_mongo_admin()
        mongo = storage._db._db
        trials = mongo["trials"].find()

        assert (
            str(trials[0]["_id"]) == client_trials[0].db_id
        ), "Trial exists inside the database"


def test_observe():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        # no experiment
        with pytest.raises(ExperiementIsNotSetup):
            trials = client.suggest()

        # create an experiment
        client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        # Suggest a trial using current experiment
        trials = client.suggest()

        print(trials)
        assert len(trials) > 0
        client.observe(trials[0], [dict(name="objective", type="objective", value=1)])

        storage = get_mongo_admin()
        mongo = storage._db._db
        trials = mongo["trials"].find()

        assert trials[0].get("results") is not None, "Trial has results"
        assert trials[0]["results"] == [
            dict(name="objective", type="objective", value=1)
        ]


def test_heartbeat():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        # create an experiment
        client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        # Suggest a trial for heartbeat
        client_trials = client.suggest()
        assert len(client_trials) > 0

        storage = get_mongo_admin()
        mongo = storage._db._db
        trials = list(mongo["trials"].find())

        old_heartbeat = trials[0]["heartbeat"]

        # Update heartbeat
        client.heartbeat(client_trials[0])

        trials = list(mongo["trials"].find())
        new_heartbeat = trials[0]["heartbeat"]
        assert old_heartbeat != new_heartbeat, "Heartbeat should have changed"


def test_is_done():
    with server():
        client = ClientREST(ENDPOINT, TOKEN)

        # create an experiment
        client.new_experiment(
            name="MyExperiment",
            space=dict(a="uniform(0, 1)", b="uniform(0, 1)"),
            max_trials=10,
        )

        while not client.is_done():
            trials = client.suggest()
            client.observe(
                trials[0], [dict(name="objective", type="objective", value=1)]
            )
            print(trials[0])

        storage = get_mongo_admin()
        mongo = storage._db._db
        trials = list(mongo["trials"].find())
        assert len(trials) == 10

        print("Done")
