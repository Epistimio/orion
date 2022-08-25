"""Integration testing between the REST service and the client"""

import logging
import pytest
from bson import ObjectId

from orion.service.client import ClientREST, ExperiementIsNotSetup, RemoteException
from orion.service.testing import server, get_mongo_admin

TOKEN = "Tok1"
TOKEN2 = "Tok2"


log = logging.getLogger(__file__)


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
