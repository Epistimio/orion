"""
Trial Pacemaker
===============

Monitor trials and update their heartbeat

"""
import threading


class TrialPacemaker(threading.Thread):
    """Monitor a given trial inside a thread, updating its heartbeat
    at a given interval of time.

    Parameters
    ----------
    exp: Experiment
        The current Experiment.

    """

    def __init__(self, trial, client, wait_time=60):
        threading.Thread.__init__(self)
        self.stopped = threading.Event()
        self.trial = trial
        self.wait_time = wait_time
        self.client = client

    def stop(self):
        """Stop monitoring."""
        self.stopped.set()
        self.join()

    def run(self):
        """Run the trial monitoring every given interval."""
        while not self.stopped.wait(self.wait_time):
            self._monitor_trial()

    def _monitor_trial(self):
        # pylint: disable=protected-access
        if self.client._update_heardbeat():
            self.stopped.set()
