
from orion.service.client.experiment import ExperimentClientREST
from orion.service.testing import server


def function(a, b):
    return [dict(value=a + b, type="objective", name="whatever")]


TOKEN1 = "Tok1"


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
        working_dir=None,
        debug=False,
    )
    return client


def test_client_actions():
    """Test client actions"""
    
    with server() as (endpoint, _):
        client = new_client(endpoint, TOKEN1)

        inserted_trial = client.insert({"a": 0.5, "b": 0.5})
        
        assert inserted_trial is not None
        
        assert len(client.fetch_trials()) == 1
        assert len(client.fetch_pending_trials()) == 1
        assert len(client.fetch_trials_by_status("new")) == 1
        
        # this includes reserved trials
        assert len(client.fetch_noncompleted_trials()) == 1
        
        print(inserted_trial)
        assert client.get_trial(inserted_trial) is not None
        
        
        
        

