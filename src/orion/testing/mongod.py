import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Manager, Process

log = logging.getLogger(__file__)


def launch_mongod(shared, port, address, dir) -> None:
    """Execute the mongoDB server in a subprocess."""
    arguments = [
        "--dbpath",
        dir,
        "--wiredTigerCacheSizeGB",
        "1",
        "--port",
        str(port),
        "--bind_ip",
        address,
        "--pidfilepath",
        os.path.join(dir, "pid"),
    ]

    kwargs = dict(
        args=" ".join(["mongod"] + arguments),
        stdout=subprocess.PIPE,
        bufsize=1,
        stderr=subprocess.STDOUT,
    )

    shared["db_pid_path"] = os.path.join(dir, "pid")

    with subprocess.Popen(**kwargs, shell=True) as proc:
        try:
            shared["pid"] = proc.pid

            if "dp_pid" not in shared:
                try:
                    shared["dp_pid"] = int(open(os.path.join(dir, "pid")).read())
                    shared["running"] = True
                except FileNotFoundError:
                    pass

            while shared["running"]:
                if proc.poll() is None:
                    line = proc.stdout.readline().decode("utf-8")[:-1]
                    log.debug(line)
                else:
                    shared["running"] = False
                    shared["exit"] = proc.returncode
                    log.debug("Stopping mongod popen")

        except Exception:
            log.error(traceback.format_exc())
            shared["running"] = False
            return


@contextmanager
def process_mongod(port, address) -> None:
    """Launch a mongoDB server in parallel. The server is stop on exit"""

    with tempfile.TemporaryDirectory() as dir:
        with Manager() as manager:
            shared = manager.dict()
            shared["running"] = False

            proc = Process(target=launch_mongod, args=(shared, port, address, dir))
            proc.start()

            acc = 0
            while not shared["running"] and acc < 2:
                acc += 0.01
                time.sleep(0.01)

            log.debug("Mongod ready after %f", acc)

            yield proc

            log.debug("Should stop")

            shared["running"] = False
            stop_mongodb(dir)

            log.debug("Stopping mongod process")
            shutil.rmtree(dir)
            proc.kill()


def start_mongodb(port, address, dir):
    """Start the mongoDB server."""

    os.environ["MONGO_PATH"] = dir
    os.environ["MONGO_PORT"] = str(port)
    os.environ["MONGO_ADDRESS"] = str(address)
    mongodb_run_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "setup.sh")
    )

    log.debug("Started mongo")
    subprocess.call(
        args=["bash", mongodb_run_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_mongodb(dir):
    """Stop the mongoDB server."""
    subprocess.run(["mongod", "--shutdown", "--dbpath", dir])


@dataclass
class MongoDBConfig:
    port: int
    address: str
    dir: str
    admin: str = "god"
    admin_password: str = "god123"


@contextmanager
def mongod(port, address) -> None:
    """Launch a mongoDB server in parallel. The server is stop on exit"""
    log.debug("Starting mongodb on %s:%d", address, port)
    with tempfile.TemporaryDirectory() as dir:
        start_mongodb(port, address, dir)

        yield MongoDBConfig(port, address, dir)

        stop_mongodb(dir)


def _add_mongo_user(mongo, user, password, db="amin", role="admin"):
    """Add a user to the mongoDB server."""

    create_user = dict(
        user=user,
        pwd=password,
        roles=[
            dict(role=role, db=db),
        ],
    )

    subprocess.run(
        [
            "mongo",
            f"{mongo.address}:{str(mongo.port)}",
            "--authenticationDatabase",
            "admin",
            "-u",
            mongo.admin,
            "-p",
            mongo.admin_password,
            "--eval",
            f"db.createUser({json.dumps(create_user)})",
        ]
    )


def add_admin_user(mongo, user, password):
    """Add a admin user to the mongoDB server."""
    _add_mongo_user(mongo, user, password, db="admin", role="userAdminAnyDatabase")


def add_orion_user(mongo, user, password):
    """Add a orion user to the mongoDB server."""
    _add_mongo_user(mongo, user, password, db="orion", role="readWrite")


def main():
    """Entry point for tox to setup and shutdown testing databases"""

    parser = argparse.ArgumentParser(description="Setup and shutdown testing databases")
    parser.add_argument(
        "command", type=str, choices=["start", "stop"], help="Command to execute"
    )
    parser.add_argument(
        "--port", type=int, default=27017, help="Port for the mongoDB server"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="localhost",
        help="Address for the mongoDB server",
    )
    parser.add_argument(
        "--dir", type=str, default="/tmp/orion", help="Directory for the mongoDB server"
    )
    args = parser.parse_args()

    if args.command == "start":
        start_mongodb(args.port, args.address, args.dir)
    else:
        stop_mongodb(args.dir)
