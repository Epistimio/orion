"""Tests for storage resource

NB: It seems ephemeral db cannot be shared across processes.
It could still be used to test /dump, but we need a database than can be
managed from many processes to test /load, as /load launches import task
in a separate process.

So, we instead use a PickledDB as destination database for /load testings.
"""
import os
import random
import string
import time

from falcon import testing

from orion.core.io.database.pickleddb import PickledDB
from orion.core.worker.storage_backup import load_database
from orion.serving.storage_resource import _gen_host_file
from orion.serving.webapi import WebApi
from orion.storage.base import setup_storage

LOAD_DATA = os.path.join(
    os.path.dirname(__file__), "..", "commands", "orion_db_load_test_data.pickled"
)


def _load_data(ephemeral_storage):
    """Load test data in ephemeral storage. To be used before testing /dump requests."""
    assert os.path.isfile(LOAD_DATA)
    load_database(ephemeral_storage.storage, LOAD_DATA, resolve="ignore")


def _clean_dump(dump_path):
    """Delete dumped files."""
    for path in (dump_path, f"{dump_path}.lock"):
        if os.path.isfile(path):
            os.unlink(path)


def _random_string(length: int):
    """Generate a random string with given length. Used to generate multipart form data for /load request"""
    domain = string.ascii_lowercase + string.digits
    return "".join(random.choice(domain) for _ in range(length))


def _gen_multipart_form_form_load(file: str, resolve: str, name="", version=""):
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


def test_dump_all(client, ephemeral_storage):
    """Test simple call to /dump"""
    _load_data(ephemeral_storage)
    response = client.simulate_get("/dump")
    host = _gen_host_file()
    try:
        with open(host, "wb") as file:
            file.write(response.content)
        dumped_db = PickledDB(host)
        with dumped_db.locked_database(write=False) as internal_db:
            collections = set(internal_db._db.keys())
        assert collections == {"experiments", "algo", "trials", "benchmarks"}
        assert len(dumped_db.read("experiments")) == 3
        assert len(dumped_db.read("algo")) == 3
        assert len(dumped_db.read("trials")) == 24
        assert len(dumped_db.read("benchmarks")) == 0
    finally:
        _clean_dump(host)


def test_dump_one_experiment(client, ephemeral_storage):
    """Test dump only experiment test_single_exp (no version specified)"""
    _load_data(ephemeral_storage)
    response = client.simulate_get("/dump?name=test_single_exp")
    host = _gen_host_file()
    try:
        with open(host, "wb") as file:
            file.write(response.content)
        dumped_db = PickledDB(host)
        assert len(dumped_db.read("benchmarks")) == 0
        experiments = dumped_db.read("experiments")
        algos = dumped_db.read("algo")
        trials = dumped_db.read("trials")
        assert len(experiments) == 1
        (exp_data,) = experiments
        # We must have dumped version 1
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 1
        assert len(algos) == len(exp_data["algorithms"]) == 1
        # This experiment must have 12 trials (children included)
        assert len(trials) == 12
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)
    finally:
        _clean_dump(host)


def test_dump_one_experiment_other_version(client, ephemeral_storage):
    """Test dump version 2 of experiment test_single_exp"""
    _load_data(ephemeral_storage)
    response = client.simulate_get("/dump?name=test_single_exp&version=2")
    host = _gen_host_file()
    try:
        with open(host, "wb") as file:
            file.write(response.content)
        dumped_db = PickledDB(host)
        assert len(dumped_db.read("benchmarks")) == 0
        experiments = dumped_db.read("experiments")
        algos = dumped_db.read("algo")
        trials = dumped_db.read("trials")
        assert len(experiments) == 1
        (exp_data,) = experiments
        # We must have dumped version 2
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 2
        assert len(algos) == len(exp_data["algorithms"]) == 1
        # This experiment must have only 6 trials
        assert len(trials) == 6
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)
    finally:
        _clean_dump(host)


def test_dump_unknown_experiment(client, ephemeral_storage):
    """Test dump unknown experiment"""
    _load_data(ephemeral_storage)
    response = client.simulate_get("/dump?name=unknown")
    assert response.status == "404 Not Found"
    assert response.json == {
        "title": "DatabaseError",
        "description": "No experiment found with query {'name': 'unknown'}. Nothing to dump.",
    }


def test_load_all():
    """Test both /load and /import-status"""

    # Create empty PKL file as destination database
    host = _gen_host_file("test")
    try:
        # Setup storage and client
        pickled_storage = setup_storage(
            {"type": "legacy", "database": {"type": "PickledDB", "host": host}}
        )
        pickled_client = testing.TestClient(WebApi(pickled_storage, {}))
        # Retrieve database
        dst_db = pickled_storage._db

        # Make sure database is empty
        assert len(dst_db.read("benchmarks")) == 0
        assert len(dst_db.read("experiments")) == 0
        assert len(dst_db.read("trials")) == 0
        assert len(dst_db.read("algo")) == 0

        # Generate body and header for request /load
        body, headers = _gen_multipart_form_form_load(LOAD_DATA, "ignore")

        # Test /load and /import-status 5 times with resolve=ignore
        # to check if data are effectively ignored on conflict
        for _ in range(5):
            task = pickled_client.simulate_post(
                "/load", headers=headers, body=body
            ).json["task"]
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
            assert all(
                message.startswith("INFO:orion.core.worker.storage_backup")
                for message in messages
            )
            # Check final task status
            assert progress["status"] == "finished"
            assert progress["progress_value"] == 1.0

            # Check expected data count in database
            # Count should not change as data are ignored on conflict every time
            assert len(dst_db.read("benchmarks")) == 0
            assert len(dst_db.read("experiments")) == 3
            assert len(dst_db.read("trials")) == 24
            assert len(dst_db.read("algo")) == 3

    finally:
        _clean_dump(host)


def test_load_one_experiment():
    """Test both /load and /import-status for one experiment"""

    # Create empty PKL file as destination database
    host = _gen_host_file("test")
    try:
        # Setup storage and client
        pickled_storage = setup_storage(
            {"type": "legacy", "database": {"type": "PickledDB", "host": host}}
        )
        pickled_client = testing.TestClient(WebApi(pickled_storage, {}))
        # Retrieve database
        dst_db = pickled_storage._db

        # Make sure database is empty
        assert len(dst_db.read("benchmarks")) == 0
        assert len(dst_db.read("experiments")) == 0
        assert len(dst_db.read("trials")) == 0
        assert len(dst_db.read("algo")) == 0

        # Generate body and header for request /load
        body, headers = _gen_multipart_form_form_load(
            LOAD_DATA, "ignore", "test_single_exp"
        )

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
        assert all(
            message.startswith("INFO:orion.core.worker.storage_backup")
            for message in messages
        )
        # Check final task status
        assert progress["status"] == "finished"
        assert progress["progress_value"] == 1.0

        # Check expected data count in database
        experiments = dst_db.read("experiments")
        algos = dst_db.read("algo")
        trials = dst_db.read("trials")

        assert len(experiments) == 1
        (exp_data,) = experiments
        # We must have dumped version 1
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 1
        assert len(algos) == len(exp_data["algorithms"]) == 1
        # This experiment must have 12 trials (children included)
        assert len(trials) == 12
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)

    finally:
        _clean_dump(host)


def test_load_one_experiment_other_version():
    """Test both /load and /import-status for one experiment with specific version"""

    # Create empty PKL file as destination database
    host = _gen_host_file("test")
    try:
        # Setup storage and client
        pickled_storage = setup_storage(
            {"type": "legacy", "database": {"type": "PickledDB", "host": host}}
        )
        pickled_client = testing.TestClient(WebApi(pickled_storage, {}))
        # Retrieve database
        dst_db = pickled_storage._db

        # Make sure database is empty
        assert len(dst_db.read("benchmarks")) == 0
        assert len(dst_db.read("experiments")) == 0
        assert len(dst_db.read("trials")) == 0
        assert len(dst_db.read("algo")) == 0

        # Generate body and header for request /load
        body, headers = _gen_multipart_form_form_load(
            LOAD_DATA, "ignore", "test_single_exp", "2"
        )

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
        assert all(
            message.startswith("INFO:orion.core.worker.storage_backup")
            for message in messages
        )
        # Check final task status
        assert progress["status"] == "finished"
        assert progress["progress_value"] == 1.0

        # Check expected data count in database
        experiments = dst_db.read("experiments")
        algos = dst_db.read("algo")
        trials = dst_db.read("trials")

        assert len(experiments) == 1
        (exp_data,) = experiments
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 2
        assert len(algos) == len(exp_data["algorithms"]) == 1
        # This experiment must have only 6 trials
        assert len(trials) == 6
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)

    finally:
        _clean_dump(host)


def test_load_unknown_experiment():
    """Test both /load and /import-status for an unknown experiment"""

    # Create empty PKL file as destination database
    host = _gen_host_file("test")
    try:
        # Setup storage and client
        pickled_storage = setup_storage(
            {"type": "legacy", "database": {"type": "PickledDB", "host": host}}
        )
        pickled_client = testing.TestClient(WebApi(pickled_storage, {}))
        # Retrieve database
        dst_db = pickled_storage._db

        # Make sure database is empty
        assert len(dst_db.read("benchmarks")) == 0
        assert len(dst_db.read("experiments")) == 0
        assert len(dst_db.read("trials")) == 0
        assert len(dst_db.read("algo")) == 0

        # Generate body and header for request /load
        body, headers = _gen_multipart_form_form_load(LOAD_DATA, "ignore", "unknown")

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
        # Check final task status (error expected, progress not finished)
        assert progress["status"] == "error"
        assert progress["progress_value"] < 1.0
        # Check we have messages
        # Last message must be an error message
        assert messages
        assert all(
            message.startswith("INFO:orion.core.worker.storage_backup")
            for message in messages[:-1]
        )
        assert (
            messages[-1]
            == "Error: No experiment found with query {'name': 'unknown'}. Nothing to import."
        )

        # Check expected data count in database (must be still empty)
        assert len(dst_db.read("experiments")) == 0
        assert len(dst_db.read("algo")) == 0
        assert len(dst_db.read("trials")) == 0

    finally:
        _clean_dump(host)
