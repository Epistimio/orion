from orion.service.client.workon import BaseClientREST


class StorageClientREST(BaseClientREST):
    def __init__(self, endpoint, token) -> None:
        super().__init__(endpoint, token)

    def insert(self, params, results=None, reserve=False):
        payload = self._post(
            "insert",
            experiment_name=self.experiment_name,
            params=params,
            results=results,
            reserve=reserve,
        )

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_noncompleted_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )

    def fetch_pending_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_pending_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        payload = self._post(
            "fetch_trials_by_status",
            experiment_name=self.experiment_name,
            status=status,
            with_evc_tree=with_evc_tree,
        )

    def get_trial(self, trial=None, uid=None):
        payload = self._post(
            "get_trial",
            experiment_name=self.experiment_name,
            trial=trial,
            uid=uid,
        )

    def fetch_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )
