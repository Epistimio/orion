from orion.service.client.base import BaseClientREST, RemoteTrial


class ClientActionREST(BaseClientREST):
    def __init__(self, experiment, endpoint, token) -> None:
        super().__init__(endpoint, token)
        self.experiment = experiment

    @property
    def experiment_name(self):
        return self.experiment.name

    def insert(self, params, results=None, reserve=False):
        payload = self._post(
            "insert",
            experiment_name=self.experiment_name,
            params=params,
            results=results,
            reserve=False,
        )

        return self._to_trial(payload.get("result", None))

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_noncompleted_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )

        return self._to_trials(payload.get("result", []))

    def fetch_pending_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_pending_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )

        return self._to_trials(payload.get("result", []))

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        payload = self._post(
            "fetch_trials_by_status",
            experiment_name=self.experiment_name,
            status=status,
            with_evc_tree=with_evc_tree,
        )

        return self._to_trials(payload.get("result", []))

    def get_trial(self, trial=None, uid=None):
        payload = self._post(
            "get_trial",
            experiment_name=self.experiment_name,
            trial=trial,
            uid=uid,
        )

        return self._to_trial(payload.get("result"))

    def fetch_trials(self, with_evc_tree=False):
        payload = self._post(
            "fetch_trials",
            experiment_name=self.experiment_name,
            with_evc_tree=with_evc_tree,
        )

        return self._to_trials(payload.get("result", []))

    def _to_trial(self, trial):
        if trial is None:
            return None

        return RemoteTrial(**trial, exp_working_dir=self.experiment.working_dir)

    def _to_trials(self, results):
        if results is None:
            return []

        trials = []
        for trial in results:
            trials.append(self._to_trial(trial))

        return trials
