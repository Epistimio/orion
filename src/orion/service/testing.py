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


def wait(p):
    acc = 0
    while p.is_alive() and acc < 2:
        acc += 0.01
        time.sleep(0.01)


class AuthenticationServiceMock(AuthenticationServiceInterface):
    """Simple authentication service for testing"""

    def __init__(self, config) -> None:
        self.tok_to_user = {
            "Tok1": ("User1", "Pass1"),
            "Tok2": ("User2", "Pass2"),
            "Tok3": ("User3", "Pass3"),
        }

    def authenticate(self, token):
        username, password = self.tok_to_user.get(token, NO_CREDENTIAL)

        log.debug("Authenticated %s => %s", token, username)
        return username, password


@contextmanager
def service(port, address, servicectx) -> None:
    import time

    from orion.service.service import main

    servicectx.auth = servicectx.auth or AuthenticationServiceMock(servicectx)

    log.debug("Launching service port: %d", port)
    p = multiprocessing.Process(target=main, args=(address, port, servicectx))
    p.start()

    # The server takes a bit of time to setup
    time.sleep(1)

    try:
        yield p
    finally:
        # raise KeyboardInterrupt for regular shutdown
        os.kill(p.pid, signal.SIGINT)
        wait(p)

        if p.is_alive():
            log.debug("process still alive after sigint")
            # notify the process we want to terminate it with SIGTERM
            p.terminate()
            wait(p)

        if p.is_alive():
            log.debug("process still alive after sigterm")
            # process is taking too long kill it
            p.kill()
            wait(p)

        p.join()


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
    global MONGO_DB_PORT

    MONGO_DB_PORT, HTTP_PORT = get_free_ports(2)
    MONGO_DB_ADDRESS = "localhost"
    ENDPOINT = f"http://localhost:{HTTP_PORT}"

    servicectx = ServiceContext()
    servicectx.database.host = MONGO_DB_ADDRESS
    servicectx.database.port = MONGO_DB_PORT

    log.debug("Launching mongodb port: %d", servicectx.database.port)

    with mongod(servicectx.database.port, servicectx.database.host):
        with service(HTTP_PORT, "localhost", servicectx):
            yield ENDPOINT, MONGO_DB_PORT


def get_mongo_admin(port=MONGO_DB_PORT):
    db = MongoDB(
        name="orion",
        host="localhost",
        port=port,
        username="god",
        password="god123",
        owner=None,
    )

    return Legacy(database_instance=db, setup=False)
