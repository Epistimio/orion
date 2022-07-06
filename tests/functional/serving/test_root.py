"""Perform functional tests for the REST endpoint `/`"""


def test_runtime_summary(client):
    """Tests if the instance's meta-summary is present"""
    response = client.simulate_get("/")

    assert response.json["orion"]
    assert response.json["server"] == "gunicorn"
    assert response.json["database"] == "EphemeralDB"
    assert response.status == "200 OK"


def test_cors_with_default_frontends_uri(client):
    """Tests CORS with default frontends_uri"""

    # No origin (e.g. from a browser), should pass
    response = client.simulate_get("/")
    assert response.status == "200 OK"

    # Default origin, should pass
    response = client.simulate_get("/", headers={"Origin": "http://localhost:3000"})
    assert response.status == "200 OK"

    # Any other origin should fail (even with same host but different port)

    response = client.simulate_get("/", headers={"Origin": "http://localhost:4000"})
    assert response.status == "403 Forbidden"

    response = client.simulate_get("/", headers={"Origin": "http://localhost"})
    assert response.status == "403 Forbidden"

    response = client.simulate_get("/", headers={"Origin": "http://google.com"})
    assert response.status == "403 Forbidden"


def test_cors_with_custom_frontends_uri(client_with_frontends_uri):
    """Tests CORS with custom frontends_uri"""

    # No origin (e.g. from a browser), should pass
    response = client_with_frontends_uri.simulate_get("/")
    assert response.status == "200 OK"

    # Allowed address, should pass
    response = client_with_frontends_uri.simulate_get(
        "/", headers={"Origin": "http://123.456"}
    )
    assert response.status == "200 OK"

    # Another allowed address, should pass
    response = client_with_frontends_uri.simulate_get(
        "/", headers={"Origin": "http://example.com"}
    )
    assert response.status == "200 OK"

    # Any other origin should fail

    response = client_with_frontends_uri.simulate_get(
        "/", headers={"Origin": "http://localhost"}
    )
    assert response.status == "403 Forbidden"

    response = client_with_frontends_uri.simulate_get(
        "/", headers={"Origin": "http://google.com"}
    )
    assert response.status == "403 Forbidden"
