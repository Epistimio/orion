# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.trial_monitor` -- Monitor trial execution
=================================================================
.. module:: trial_monitor
   :platform: Unix
   :synopsis: Monitor trials and update their heartbeat

"""
import datetime
import threading

from orion.core.io.database import Database


class TrialMonitor(threading.Thread):
    """Monitor a given trial inside a thread, updating its heartbeat
    at a given interval  of time.

    Parameters
    ----------
    exp: Experiment
        The current Experiment.

    """

    def __init__(self, exp, trial_id, wait_time=60):
        """Initialize a TrialMonitor."""
        threading.Thread.__init__(self)
        self.stopped = threading.Event()
        self.exp = exp
        self.trial_id = trial_id
        self.wait_time = wait_time

    def stop(self):
        """Stop monitoring."""
        self.stopped.set()
        self.join()

    def run(self):
        """Run the trial monitoring every given interval."""
        while not self.stopped.wait(self.wait_time):
            self._monitor_trial()

    def _monitor_trial(self):
        query = {'_id': self.trial_id, 'status': 'reserved'}
        trials = self.exp.fetch_trials(query)

        if trials:
            update = dict(heartbeat=datetime.datetime.utcnow())
            Database().write('trials', update, query)
        else:
            self.stopped.set()
