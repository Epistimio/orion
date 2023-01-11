"""Integration testing between the REST service and the client"""

import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager

from orion.core.io.database.mongodb import MongoDB
from orion.service.auth import NO_CREDENTIAL, AuthenticationServiceInterface
from orion.service.broker.broker import ServiceContext
from orion.storage.legacy import Legacy
from orion.testing.mongod import mongod

log = logging.getLogger(__name__)


def wait(process):
    """Wait for a process to finish"""
    acc = 0
    while process.is_alive() and acc < 2:
        acc += 0.01
        time.sleep(0.01)


# pylint: disable=too-few-public-methods
class AuthenticationServiceMock(AuthenticationServiceInterface):
    """Simple authentication service for testing"""

    # pylint: disable=super-init-not-called
    def __init__(self, config) -> None:
        self.tok_to_user = {
            "Tok1": ("User1", "Pass1"),
            "Tok2": ("User2", "Pass2"),
            "Tok3": ("User3", "Pass3"),
        }

    def authenticate(self, token):
        """Authenticate a user given its API token"""
        username, password = self.tok_to_user.get(token, NO_CREDENTIAL)

        log.debug("Authenticated %s => %s", token, username)
        return username, password


@contextmanager
def service(port, address, servicectx) -> None:
    """Launch a orion service on the given port and address"""
    from orion.service.service import main

    servicectx.auth = servicectx.auth or AuthenticationServiceMock(servicectx)

    log.debug("Launching service port: %d", port)
    proc = multiprocessing.Process(target=main, args=(address, port, servicectx))
    proc.start()

    # The server takes a bit of time to setup
    time.sleep(1)

    try:
        yield proc
    finally:
        # raise KeyboardInterrupt for regular shutdown
        os.kill(proc.pid, signal.SIGINT)
        wait(proc)

        if proc.is_alive():
            log.debug("process still alive after sigint")
            # notify the process we want to terminate it with SIGTERM
            proc.terminate()
            wait(proc)

        if proc.is_alive():
            log.debug("process still alive after sigterm")
            # process is taking too long kill it
            proc.kill()
            wait(proc)

        proc.join()


def get_free_ports(number=1):
    """Get a free port for the mongodb & the http server to allow tests in parallel"""
    import socket

    sockets = []
    ports = []

    for _ in range(number):
        sock = socket.socket()
        sock.bind(("", 0))
        ports.append(sock.getsockname()[1])
        sockets.append(sock)

    for sock in sockets:
        sock.close()

    return tuple(ports)


MONGO_DB_PORT = None


@contextmanager
def server():
    """Launch a new mongodb server and an orion service for testing"""
    # pylint: disable=global-statement
    global MONGO_DB_PORT

    # pylint: disable=unbalanced-tuple-unpacking
    mongodb_port, http_port = get_free_ports(2)
    MONGO_DB_PORT = mongodb_port

    mongodb_address = "localhost"
    endpoint = f"http://localhost:{http_port}"

    servicectx = ServiceContext()
    servicectx.database.host = mongodb_address
    servicectx.database.port = mongodb_port

    log.debug("Launching mongodb port: %d", servicectx.database.port)

    with mongod(servicectx.database.port, servicectx.database.host):
        with service(http_port, "localhost", servicectx):
            yield endpoint, mongodb_port


def get_mongo_admin(port=MONGO_DB_PORT, owner=None):
    """Return an admin connection to a mongodb connection"""
    db = MongoDB(
        name="orion",
        host="localhost",
        port=port,
        username="god",
        password="god123",
        owner=owner,
    )

    return Legacy(database_instance=db, setup=False)
