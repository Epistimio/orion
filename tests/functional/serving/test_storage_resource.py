"""Tests for storage resource

NB: It seems ephemeral db cannot be shared across processes.
It could still be used to test /dump, but we need a database than can be
managed from many processes to test /load, as /load launches import task
in a separate process.

So, we instead use a PickledDB as destination database for /load testings.
"""
import logging
import os
import random
import string
import time

import pytest
from falcon import testing

from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils import generate_temporary_file
from orion.serving.webapi import WebApi
from orion.storage.backup import load_database
from orion.storage.base import setup_storage


@pytest.fixture
def ephemeral_loaded(ephemeral_storage, pkl_experiments):
    """Load test data in ephemeral storage. To be used before testing /dump requests."""
    load_database(ephemeral_storage.storage, pkl_experiments, resolve="ignore")


@pytest.fixture
def ephemeral_loaded_with_benchmarks(ephemeral_storage, pkl_experiments_and_benchmarks):
    """Load test data in ephemeral storage. To be used before testing /dump requests."""
    load_database(
        ephemeral_storage.storage, pkl_experiments_and_benchmarks, resolve="ignore"
    )


class DumpContext:
    def __init__(self, client, parameters=None):
        self.client = client
        self.host = generate_temporary_file()
        self.db = None
        self.url = "/dump" + ("" if parameters is None else f"?{parameters}")

    def __enter__(self):
        response = self.client.simulate_get(self.url)
        with open(self.host, "wb") as file:
            file.write(response.content)
        self.db = PickledDB(self.host)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _clean_dump(self.host)
        print("CLEANED DUMP")


class LoadContext:
    def __init__(self):
        # Create empty PKL file as destination database
        self.host = generate_temporary_file("test")
        # Setup storage and client
        pickled_storage = setup_storage(
            {"type": "legacy", "database": {"type": "PickledDB", "host": self.host}}
        )
        self.pickled_client = testing.TestClient(WebApi(pickled_storage, {}))
        self.db = pickled_storage._db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _clean_dump(self.host)
        print("CLEANED LOAD")


def _clean_dump(dump_path):
    """Delete dumped files."""
    for path in (dump_path, f"{dump_path}.lock"):
        if os.path.isfile(path):
            os.unlink(path)


def _random_string(length: int):
    """Generate a random string with given length. Used to generate multipart form data for /load request"""
    domain = string.ascii_lowercase + string.digits
    return "".join(random.choice(domain) for _ in range(length))


def _gen_multipart_form_for_load(file: str, resolve: str, name="", version=""):
    """Generate multipart form body and headers for /load request with uploaded file.
    Help (2022/11/08):
    https://stackoverflow.com/a/45703104
    https://stackoverflow.com/a/23517227
    """
    import io

    boundary = "----MultiPartFormBoundary" + _random_string(16)
    in_boundary = "--" + boundary
    out_boundary = in_boundary + "--"

    buff = io.BytesIO()

    buff.write(in_boundary.encode())
    buff.write(b"\r\n")
    buff.write(b'Content-Disposition: form-data; name="file"; filename="load.pkl"')
    buff.write(b"\r\n")
    buff.write(b"Content-Type: application/octet-stream")
    buff.write(b"\r\n")
    buff.write(b"\r\n")
    with open(file, "rb") as data:
        buff.write(data.read())
    buff.write(b"\r\n")

    buff.write(in_boundary.encode())
    buff.write(b"\r\n")
    buff.write(b'Content-Disposition: form-data; name="resolve"')
    buff.write(b"\r\n")
    buff.write(b"\r\n")
    buff.write(resolve.encode())
    buff.write(b"\r\n")

    buff.write(in_boundary.encode())
    buff.write(b"\r\n")
    buff.write(b'Content-Disposition: form-data; name="name"')
    buff.write(b"\r\n")
    buff.write(b"\r\n")
    buff.write(name.encode())
    buff.write(b"\r\n")

    buff.write(in_boundary.encode())
    buff.write(b"\r\n")
    buff.write(b'Content-Disposition: form-data; name="version"')
    buff.write(b"\r\n")
    buff.write(b"\r\n")
    buff.write(version.encode())
    buff.write(b"\r\n")

    buff.write(out_boundary.encode())
    buff.write(b"\r\n")

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(buff.tell()),
    }

    return buff.getvalue(), headers


def test_dump_all(client, ephemeral_loaded_with_benchmarks, testing_helpers):
    """Test simple call to /dump"""
    with DumpContext(client) as ctx:
        testing_helpers.assert_tested_db_structure(ctx.db)


def test_dump_one_experiment(client, ephemeral_loaded_with_benchmarks, testing_helpers):
    """Test dump only experiment test_single_exp (no version specified)"""
    with DumpContext(client, "name=test_single_exp") as ctx:
        # We must have dumped version 2
        testing_helpers.check_unique_import_test_single_expV2(ctx.db)


def test_dump_one_experiment_other_version(
    client, ephemeral_loaded_with_benchmarks, testing_helpers
):
    """Test dump version 1 of experiment test_single_exp"""
    with DumpContext(client, "name=test_single_exp&version=1") as ctx:
        testing_helpers.check_unique_import_test_single_expV1(ctx.db)


def test_dump_unknown_experiment(client, ephemeral_loaded_with_benchmarks):
    """Test dump unknown experiment"""
    response = client.simulate_get("/dump?name=unknown")
    assert response.status == "404 Not Found"
    assert response.json == {
        "title": "DatabaseError",
        "description": "No experiment found with query {'name': 'unknown'}. Nothing to dump.",
    }


def _check_load_and_import_status(
    pickled_client, headers, body, finished=True, latest_message=None
):
    """Test both /load and /import-status"""
    task = pickled_client.simulate_post("/load", headers=headers, body=body).json[
        "task"
    ]
    # Test /import-status by retrieving all task messages
    messages = []
    while True:
        progress = pickled_client.simulate_get(f"/import-status/{task}").json
        messages.extend(progress["messages"])
        if progress["status"] != "active":
            break
        time.sleep(0.010)
    # Check we have messages
    assert messages
    for message in messages if latest_message is None else messages[:-1]:
        assert message.startswith("INFO:orion.storage.backup")
    # Check final task status
    if finished:
        assert progress["status"] == "finished"
        assert progress["progress_value"] == 1.0
    else:
        assert progress["status"] == "error"
        assert progress["progress_value"] < 1.0
    if latest_message is not None:
        assert messages[-1] == latest_message


def test_load_all(pkl_experiments_and_benchmarks, testing_helpers, caplog):
    """Test both /load and /import-status"""
    with caplog.at_level(logging.INFO):
        with LoadContext() as ctx:
            # Make sure database is empty
            testing_helpers.check_empty_db(ctx.db)

            # Generate body and header for request /load
            body, headers = _gen_multipart_form_for_load(
                pkl_experiments_and_benchmarks, "ignore"
            )

            # Test /load and /import-status 5 times with resolve=ignore
            # to check if data are effectively ignored on conflict
            for _ in range(5):
                _check_load_and_import_status(ctx.pickled_client, headers, body)
                # Check expected data in database
                # Count should not change as data are ignored on conflict every time
                testing_helpers.assert_tested_db_structure(ctx.db)


def test_load_one_experiment(pkl_experiments, testing_helpers, caplog):
    """Test both /load and /import-status for one experiment"""
    with caplog.at_level(logging.INFO):
        with LoadContext() as ctx:
            # Make sure database is empty
            testing_helpers.check_empty_db(ctx.db)

            # Generate body and header for request /load
            body, headers = _gen_multipart_form_for_load(
                pkl_experiments, "ignore", "test_single_exp"
            )

            _check_load_and_import_status(ctx.pickled_client, headers, body)

            # Check expected data in database
            # We must have loaded version 2
            testing_helpers.check_unique_import_test_single_expV2(ctx.db)


def test_load_one_experiment_other_version(
    pkl_experiments_and_benchmarks, testing_helpers, caplog
):
    """Test both /load and /import-status for one experiment with specific version"""
    with caplog.at_level(logging.INFO):
        with LoadContext() as ctx:
            # Make sure database is empty
            testing_helpers.check_empty_db(ctx.db)

            # Generate body and header for request /load
            body, headers = _gen_multipart_form_for_load(
                pkl_experiments_and_benchmarks, "ignore", "test_single_exp", "1"
            )

            _check_load_and_import_status(ctx.pickled_client, headers, body)

            # Check expected data in database
            testing_helpers.check_unique_import_test_single_expV1(ctx.db)


def test_load_unknown_experiment(pkl_experiments, testing_helpers, caplog):
    """Test both /load and /import-status for an unknown experiment"""
    with caplog.at_level(logging.INFO):
        with LoadContext() as ctx:
            # Make sure database is empty
            testing_helpers.check_empty_db(ctx.db)

            # Generate body and header for request /load
            body, headers = _gen_multipart_form_for_load(
                pkl_experiments, "ignore", "unknown"
            )

            _check_load_and_import_status(
                ctx.pickled_client,
                headers,
                body,
                finished=False,
                latest_message="Error: No experiment found with query {'name': 'unknown'}. Nothing to import.",
            )

            # Check database (must be still empty)
            testing_helpers.check_empty_db(ctx.db)


@pytest.mark.parametrize(
    "log_level,expected_message_prefixes",
    [
        (logging.WARNING, []),
        (
            logging.INFO,
            [
                "INFO:orion.storage.backup:Loaded src /tmp/",
                "INFO:orion.storage.backup:Import experiment test_single_exp.1",
                "INFO:orion.storage.backup:Import experiment test_single_exp.2",
                "INFO:orion.storage.backup:Import experiment test_single_exp_child.1",
            ],
        ),
    ],
)
def test_orion_serve_logging(
    pkl_experiments_and_benchmarks, log_level, expected_message_prefixes, caplog
):
    with caplog.at_level(log_level):
        with LoadContext() as ctx:
            # Generate body and header for request /load
            body, headers = _gen_multipart_form_for_load(
                pkl_experiments_and_benchmarks, "ignore"
            )
            # Request /load and get import task ID
            task = ctx.pickled_client.simulate_post(
                "/load", headers=headers, body=body
            ).json["task"]
            # Collect task messages using request /import-status
            messages = []
            while True:
                progress = ctx.pickled_client.simulate_get(
                    f"/import-status/{task}"
                ).json
                messages.extend(progress["messages"])
                if progress["status"] != "active":
                    break
                time.sleep(0.010)

            # Check messages
            assert len(messages) == len(expected_message_prefixes)
            for given_msg, expected_msg_prefix in zip(
                messages, expected_message_prefixes
            ):
                assert given_msg.startswith(expected_msg_prefix)
