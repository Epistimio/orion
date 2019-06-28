from orion.core.utils import Factory


class BaseStorageProtocol:
    def create_trial(self, trial):
        """create a new trial to be executed"""
        raise NotImplementedError()

    def select_trial(self, *args, **kwargs):
        """select a pending trial to be ran"""
        raise NotImplementedError()

    def fetch_completed_trials(self):
        """fetch the latest completed trials of this experiment that were not yet observed"""
        raise NotImplementedError()

    def is_done(self, experiment):
        """check if we have reached the maximum number of trials.
            This only takes into account completed trials.
            So if trials do not complete this may never return True"""
        raise NotImplementedError()

    def push_completed_trial(self, trial):
        """Process a trial and set it as a completed
            This also registers some statistics for the experiment (best trial, runtime etc...)"""
        raise NotImplementedError()

    def mark_as_broken(self, trial):
        """"When a trial fails we set it as broken. It means it will to be re ran again."""
        raise NotImplementedError()

    def get_stats(self):
        """get the statistics for the current experiment (best trial, runtime etc...)"""
        raise NotImplementedError()

    def update_trial(self, trial, **kwargs):
        """try to read the result of the trial and update the trial.results attribute """
        raise NotImplementedError()

    def get_trial(self, trial):
        """fetch a trial inside storage using its id"""
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
        MongoDB, track, cometML, MLFLow, etc...

        Protocol('track', uri='file://orion_test.json')
        Protocol('legacy', experiment=experiment)"""
    pass
