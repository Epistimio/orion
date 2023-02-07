"""
Module responsible for storage import/export REST endpoints
===========================================================

Serves all the requests made to storage import/export REST endpoints.

"""
import json
import logging
import multiprocessing
import os
import uuid
from datetime import datetime
from queue import Empty

import falcon
from falcon import Request, Response

from orion.core.io.database import DatabaseError
from orion.core.worker.storage_backup import dump_database, load_database

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _gen_host_file(basename="dump"):
    """Generate a temporary file where data could be saved.

    Create an empty file without collision in working directory.
    Return name of generated file.
    """
    index = 0
    while True:
        file_name = f"{basename}{index}.pkl"
        try:
            with open(file_name, "x"):
                return file_name
        except FileExistsError:
            index += 1
            continue


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
    """Wrapper to represent an import task.

    Properties:
    - task_id: task ID, used to identify the task in web API
    - notifications: stream handler with shared queue to capture task messages
    - completed: shared status: 0 for running, -1 for failure, 1 for success
    """

    # String representation of task status
    IMPORT_STATUS = {0: "run", -1: "fail", 1: "success"}

    def __init__(self):
        self.task_id = str(uuid.uuid4())
        self.notifications = Notifications()
        self.completed = multiprocessing.Value("i", 0)

    def is_completed(self):
        """Return True if task is completed"""
        return self.completed.value

    def set_completed(self, success=True):
        """Set task terminated status"""
        self.completed.value = 1 if success else -1


def _import_data(task: ImportTask, storage, load_host, resolve, name, version):
    """Function to run import task.

    Set stream handler, launch load_database and set task status.
    """
    try:
        print("Import starting.", task.task_id)
        logging.basicConfig(stream=task.notifications, force=True)
        load_database(storage, load_host, resolve, name, version)
        task.set_completed(success=True)
    except Exception as exc:
        print("Import error.", exc)
        # Add error message to shared messages
        task.notifications.write(f"Error: {exc}")
        task.set_completed(success=False)
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
        dump_host = _gen_host_file(basename="dump")
        download_suffix = "" if name is None else f" {name}"
        if download_suffix and version is not None:
            download_suffix = f"{download_suffix}.{version}"
        try:
            dump_database(self.storage, dump_host, name=name, version=version)
            resp.downloadable_as = f"dump{download_suffix if download_suffix else ''} ({datetime.now()}).pkl"
            resp.content_type = "application/octet-stream"
            with open(dump_host, "rb") as file:
                resp.data = file.read()
        except DatabaseError as exc:
            raise falcon.HTTPNotFound(title=type(exc).__name__, description=str(exc))
        finally:
            os.unlink(dump_host)
            os.unlink(f"{dump_host}.lock")

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
                    load_host = _gen_host_file(basename="load")
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
        latest_messages = []
        while True:
            try:
                latest_messages.append(
                    self.current_task.notifications.queue.get_nowait()
                )
            except Empty:
                break
        resp.body = json.dumps(
            {
                "messages": latest_messages,
                "status": ImportTask.IMPORT_STATUS[self.current_task.completed.value],
            }
        )
