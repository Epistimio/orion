# -*- coding: utf-8 -*-
"""
Trial Pacemaker
===============

Monitor trials and update their heartbeat

"""
import threading

from orion.storage.base import get_storage

STOPPED_STATUS = {"completed", "interrupted", "suspended"}


class TrialPacemaker(threading.Thread):
    """Monitor a given trial inside a thread, updating its heartbeat
    at a given interval of time.

    Parameters
    ----------
    exp: Experiment
        The current Experiment.

    """

    def __init__(self, trial, wait_time=60):
        threading.Thread.__init__(self)
        self.stopped = threading.Event()
        self.trial = trial
        self.wait_time = wait_time
        self.storage = get_storage()

    def stop(self):
        """Stop monitoring."""
        self.stopped.set()
        self.join()

    def run(self):
        """Run the trial monitoring every given interval."""
        while not self.stopped.wait(self.wait_time):
            self._monitor_trial()

    def _monitor_trial(self):
        trial = self.storage.get_trial(self.trial)

        if trial.status in STOPPED_STATUS:
            self.stopped.set()
        else:
            if not self.storage.update_heartbeat(trial):
                self.stopped.set()
