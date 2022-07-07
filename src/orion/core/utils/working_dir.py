"""
ContextManager for working directory
====================================

ContextManager class to create a permanent directory or a temporary one.

"""
import logging
import os
import tempfile

log = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class SetupWorkingDir:
    """ContextManager class for temporary or permanent directory.

    Parameters
    ----------
    experiment: ``orion.client.experiment.ExperimentClient``
        Experiment for which the working directory will be created

    """

    def __init__(self, experiment):
        self.experiment = experiment
        self.tmp = None
        self._tmpdir = None

    def __enter__(self):
        """Create the a permanent directory or a temporary one."""

        self.tmp = bool(not self.experiment.working_dir)

        if self.tmp:
            base_path = os.path.join(tempfile.gettempdir(), "orion")
            os.makedirs(base_path, exist_ok=True)
            self._tmpdir = tempfile.TemporaryDirectory(
                prefix=f"{self.experiment.name}-v{self.experiment.version}",
                dir=self.experiment.working_dir,
            )
            self.experiment.working_dir = self._tmpdir.name
        else:
            os.makedirs(self.experiment.working_dir, exist_ok=True)

        log.debug("Working directory at '%s':", self.experiment.working_dir)

        return self.experiment.working_dir

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup temporary directory."""
        if self.tmp:
            self._tmpdir.cleanup()
            self.experiment.working_dir = None
