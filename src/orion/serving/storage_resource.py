"""
Module responsible for storage import/export REST endpoints
===========================================================

Serves all the requests made to storage import/export REST endpoints.

"""
import json
import logging
import multiprocessing
import os
import traceback
import uuid
from datetime import datetime
from queue import Empty

import falcon
from falcon import Request, Response

from orion.core.io.database import DatabaseError
from orion.core.utils import generate_temporary_file
from orion.storage.backup import dump_database, load_database


class Notifications:
    """Stream handler to collect messages in a shared queue.

    Used instead of stdout in import progress to capture log messages.
    """

    def __init__(self):
        """Initialize with a shared queue."""
        self.queue = multiprocessing.Queue()

    def write(self, buf: str):
        """Write received data"""
        for line in buf.rstrip().splitlines():
            self.queue.put(line)

    def flush(self):
        """Placeholder to flush data"""


class ImportTask:
    """Wrapper to represent an import task. Used to monitor task progress.

    There is two ways to monitor task:

    - either get messages collected in stream handler queue.
      Stream handler collects all messages logged in task.
    - either regularly check latest message in progress_message.
      Progress message only describe the latest step running in task.

    Attributes
    ----------
    task_id: str
        Used to identify the task in web API
    _notifications: Notifications
        Stream handler with shared queue to capture task messages
    _progress_message:
        Latest progress message
    _progress_value:
        Latest progress (0 <= floating value <= 1)
    _completed:
        Shared status: 0 for running, -1 for failure, 1 for success
    _lock:
        Lock to use to prevent concurrent executions
        when updating task state.
    """

    # String representation of task status
    IMPORT_STATUS = {0: "active", -1: "error", 1: "finished"}

    def __init__(self):
        self.task_id = str(uuid.uuid4())
        self._notifications = Notifications()
        self._progress_message = multiprocessing.Array("c", 512)
        self._progress_value = multiprocessing.Value("d", 0.0)
        self._completed = multiprocessing.Value("i", 0)
        self._lock = multiprocessing.Lock()

    def set_progress(self, message: str, progress: float):
        with self._lock:
            self._progress_message.value = message.encode()
            self._progress_value.value = progress

    def is_completed(self):
        """Return True if task is completed"""
        return self._completed.value

    def set_completed(self, success=True, notification=None):
        """Set task terminated status

        Parameters
        ----------
        success: bool
            True if task is successful, False otherwise
        notification: str
            Optional message to add in notifications
        """
        with self._lock:
            self._completed.value = 1 if success else -1
            if notification:
                self._notifications.write(notification)

    def listen_logging(self):
        """Set notifications as logging stream to collect logging messages."""
        # logging.basicConfig() won't do anything if there are already handlers
        # for root logger, so we must clear previous handlers first
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers:
                handler.close()
            root_logger.handlers.clear()
        # Then set stream and keep previous log level
        logging.basicConfig(stream=self._notifications, level=root_logger.level)

    def flush_state(self):
        """Return a dictionary with current task state.

        NB: Collect all messages currently in stream handler queue,
        so stream handler queue is emptied after this method is called.
        """
        latest_messages = []
        with self._lock:
            while True:
                try:
                    latest_messages.append(self._notifications.queue.get_nowait())
                except Empty:
                    break
            return {
                "messages": latest_messages,
                "progress_message": self._progress_message.value.decode(),
                "progress_value": self._progress_value.value,
                "status": ImportTask.IMPORT_STATUS[self._completed.value],
            }


def _import_data(task: ImportTask, storage, load_host, resolve, name, version):
    """Function to run import task.

    Set stream handler, launch load_database and set task status.
    """
    try:
        print("Import starting.", task.task_id)
        task.listen_logging()
        load_database(
            storage,
            load_host,
            resolve,
            name,
            version,
            progress_callback=task.set_progress,
        )
        task.set_completed(success=True)
    except Exception as exc:
        traceback.print_tb(exc.__traceback__)
        print("Import error.", exc)
        # Add error message to shared messages
        task.set_completed(success=False, notification=f"Error: {exc}")
    finally:
        # Remove imported files
        os.unlink(load_host)
        lock_file = f"{load_host}.lock"
        if os.path.exists(lock_file):
            os.unlink(lock_file)
        print("Import terminated.")


class StorageResource:
    """Handle requests for the dump/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage
        self.current_task: ImportTask = None

    def on_get_dump(self, req: Request, resp: Response):
        """Handle the GET requests for dump/"""
        name = req.get_param("name")
        version = req.get_param_as_int("version")
        dump_host = generate_temporary_file(basename="dump")
        download_suffix = "" if name is None else f" {name}"
        if download_suffix and version is not None:
            download_suffix = f"{download_suffix}.{version}"
        try:
            dump_database(
                self.storage, dump_host, name=name, version=version, overwrite=True
            )
            resp.downloadable_as = f"dump{download_suffix if download_suffix else ''} ({datetime.now()}).pkl"
            resp.content_type = "application/octet-stream"
            with open(dump_host, "rb") as file:
                resp.data = file.read()
        except DatabaseError as exc:
            raise falcon.HTTPNotFound(title=type(exc).__name__, description=str(exc))
        finally:
            # Clean dumped files
            for path in (dump_host, f"{dump_host}.lock"):
                if os.path.exists(path):
                    os.unlink(path)

    def on_post_load(self, req: Request, resp: Response):
        """Handle the POST requests for load/"""
        if self.current_task is not None and not self.current_task.is_completed():
            raise falcon.HTTPForbidden(description="An import is already running")
        load_host = None
        resolve = None
        name = None
        version = None
        for part in req.get_media():
            if part.name == "file":
                if part.filename:
                    load_host = generate_temporary_file(basename="load")
                    with open(load_host, "wb") as dst:
                        part.stream.pipe(dst)
            elif part.name == "resolve":
                resolve = part.get_text().strip()
                if resolve not in ("ignore", "overwrite", "bump"):
                    raise falcon.HTTPInvalidParam(
                        "Invalid value for resolve", "resolve"
                    )
            elif part.name == "name":
                name = part.get_text().strip() or None
            elif part.name == "version":
                version = part.get_text().strip()
                if version:
                    try:
                        version = int(version)
                    except ValueError:
                        raise falcon.HTTPInvalidParam(
                            "Version must be an integer", "version"
                        )
                    if version < 0:
                        raise falcon.HTTPInvalidParam(
                            "Version must be a positiver integer", "version"
                        )
                else:
                    version = None
            else:
                raise falcon.HTTPInvalidParam("Unknown parameter", part.name)
        if load_host is None:
            raise falcon.HTTPInvalidParam("Missing file to import", "file")
        if resolve is None:
            raise falcon.HTTPInvalidParam("Missing resolve policy", "resolve")
        self.current_task = ImportTask()
        p = multiprocessing.Process(
            target=_import_data,
            args=(self.current_task, self.storage, load_host, resolve, name, version),
        )
        p.start()
        resp.body = json.dumps({"task": self.current_task.task_id})

    def on_get_import_status(self, req: Request, resp: Response, name: str):
        """Handle the GET requests for import-status/"""
        if self.current_task is None or self.current_task.task_id != name:
            raise falcon.HTTPInvalidParam("Unknown import task", "name")
        resp.body = json.dumps(self.current_task.flush_state())
