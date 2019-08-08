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


class TrialPacemaker(threading.Thread):
    """Monitor a given trial inside a thread, updating its heartbeat
    at a given interval of time.

    Parameters
    ----------
    exp: Experiment
        The current Experiment.

    """

    def __init__(self, exp, trial_id, wait_time=60):
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
            update = datetime.datetime.utcnow()
            if not self.exp.update_trial(trials[0], where=query, heartbeat=update):
                self.stopped.set()
        else:
            self.stopped.set()
