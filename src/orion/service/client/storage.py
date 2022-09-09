from orion.client.experiment import ExperimentClient
from orion.service.client.workon import BaseClientREST, RemoteExperiment, RemoteTrial


class StorageClientREST(BaseClientREST):
    def __init__(self, endpoint, token) -> None:
        super().__init__(endpoint, token)
