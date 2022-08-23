"""Integration testing between the REST service and the client"""

import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager

import pytest
from bson import ObjectId

from orion.core.io.database.mongodb import MongoDB
from orion.service.broker.broker import ServiceContext
from orion.service.client import ClientREST, ExperiementIsNotSetup, RemoteException
from orion.storage.legacy import Legacy
from orion.testing.mongod import mongod

TOKEN = "Tok1"
TOKEN2 = "Tok2"


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


def get_free_ports(number=1):
    """Get a free port for the mongodb & the http server to allow tests in parallel"""
    import socket

    sockets = []
    ports = []

    for _ in range(number):
        sock = socket.socket()
        sock.bind(("", 0))
        ports.append(sock.getsockname()[1])
        sockets.append(socket)

    for sock in sockets:
        sock.close()

    return tuple(ports)


MONGO_DB_PORT = None


@contextmanager
def server():
    global MONGO_DB_PORT

    MONGO_DB_PORT, HTTP_PORT = get_free_ports(2)
    MONGO_DB_ADDRESS = "localhost"
    ENDPOINT = f"http://localhost:{HTTP_PORT}"

    servicectx = ServiceContext()
    servicectx.database.host = MONGO_DB_ADDRESS
    servicectx.database.port = MONGO_DB_PORT

    with mongod(servicectx.database.port, servicectx.database.host):
        with service(HTTP_PORT, "localhost", servicectx):
            yield ENDPOINT


def get_mongo_admin(port=MONGO_DB_PORT):
    db = MongoDB(
        name="orion",
        host="localhost",
        port=port,
        username="god",
        password="god123",
        owner=None,
    )

    return Legacy(database_instance=db, setup=False)


def test_new_experiment():
    with server() as ENDPOINT:
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
    with server() as ENDPOINT:
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
        trials = list(mongo["trials"].find(dict(_id=ObjectId(client_trials[0].db_id))))

        assert len(client_trials) == 1
        assert len(trials) == 1
        assert (
            str(trials[0]["_id"]) == client_trials[0].db_id
        ), "Trial exists inside the database"


def test_observe():
    with server() as ENDPOINT:
        client = ClientREST(ENDPOINT, TOKEN)

        # no experiment
        with pytest.raises(ExperiementIsNotSetup):
            trials = client.suggest()

        # create an experiment
        client.new_experiment(
            name="MyExperiment", space=dict(a="uniform(0, 1)", b="uniform(0, 1)")
        )

        # Suggest a trial using current experiment
        client_trials = client.suggest()

        assert len(client_trials) > 0
        client.observe(
            client_trials[0], [dict(name="objective", type="objective", value=1)]
        )

        storage = get_mongo_admin()
        mongo = storage._db._db
        trials = mongo["trials"].find(dict(_id=ObjectId(client_trials[0].db_id)))

        assert trials[0].get("results") is not None, "Trial has results"
        assert trials[0]["results"] == [
            dict(name="objective", type="objective", value=1)
        ]


def test_heartbeat():
    with server() as ENDPOINT:
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
        trials = list(mongo["trials"].find(dict(_id=ObjectId(client_trials[0].db_id))))

        old_heartbeat = trials[0]["heartbeat"]

        # Update heartbeat
        client.heartbeat(client_trials[0])

        trials = list(mongo["trials"].find(dict(_id=ObjectId(client_trials[0].db_id))))
        new_heartbeat = trials[0]["heartbeat"]
        assert old_heartbeat != new_heartbeat, "Heartbeat should have changed"


def test_is_done():
    with server() as ENDPOINT:
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
        print(trials)
        assert len(trials) == 10

        print("Done")


def test_broken_experiment():
    pass


def test_authentication():
    with server() as ENDPOINT:
        client = ClientREST(ENDPOINT, "NOT A TOKEN")

        with pytest.raises(RemoteException):
            client.new_experiment(name="MyExperiment")
