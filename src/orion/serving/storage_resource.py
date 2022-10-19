"""
Module responsible for storage import/export REST endpoints
===========================================================

Serves all the requests made to storage import/export REST endpoints.

"""
import logging
import os
from datetime import datetime

import falcon
from falcon import Request, Response

from orion.core.io.database import DatabaseError
from orion.core.worker.storage_backup import dump_database

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _gen_dump_host_file():
    """Generate a temporary file where dumped data could be saved.

    Create an empty file without collision in working directory.
    Return name of generated file.
    """
    index = 0
    while True:
        file_name = f"dump{index}.pkl"
        try:
            with open(file_name, "x"):
                return file_name
        except FileExistsError:
            index += 1
            continue


class StorageResource:
    """Handle requests for the dump/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get_dump(self, req: Request, resp: Response):
        """Handle the GET requests for dump/"""
        name = req.get_param("name")
        version = req.get_param_as_int("version")
        dump_host = _gen_dump_host_file()
        download_suffix = "" if name is None else f" {name}"
        if download_suffix and version is not None:
            download_suffix = f"{download_suffix}.{version}"
        try:
            dump_database(self.storage, dump_host, experiment=name, version=version)
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
        pass
