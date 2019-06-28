from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import Factory

log = logging.getLogger(__name__)


class BaseStorageProtocol(object, metaclass=ABCMeta):
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


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """

        Protocol('track', uri='file://orion_test.json')
        Protocol('legacy', experiment=experiment)

    """
    pass
