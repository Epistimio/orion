from orion.core.utils import Factory


class BaseStorageProtocol:
    def create_trial(self, trial):
        raise NotImplementedError()

    def register_trial(self, trial):
        raise NotImplementedError()

    def select_trial(self, *args, **kwargs):
        raise NotImplementedError()

    def reserve_trial(self, *args, **kwargs):
        raise NotImplementedError()

    def fetch_completed_trials(self):
        raise NotImplementedError()

    def is_done(self, experiment):
        raise NotImplementedError()

    def push_completed_trial(self, trial):
        raise NotImplementedError()

    def mark_as_broken(self, trial):
        raise NotImplementedError()

    def get_stats(self):
        raise NotImplementedError()

    def update_trial(self, trial, **kwargs):
        raise NotImplementedError()

    def get_trial(self, trial):
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """

        Protocol('track', uri='file://orion_test.json')
        Protocol('legacy', experiment=experiment)

    """
    pass
