from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "tools" in r.json()


def test_agent_calculator():
    r = client.post("/agent", json={"tool": "calculator", "argument": "2 + 2"})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == "4"
    assert body["tool"] == "calculator"


def test_agent_calculator_bad_expression():
    r = client.post("/agent", json={"tool": "calculator", "argument": "import os"})
    assert r.status_code == 400


def test_agent_get_time():
    r = client.post("/agent", json={"tool": "get_time"})
    assert r.status_code == 200
    assert "T" in r.json()["result"]  # ISO format check


def test_agent_get_weather_known_city():
    r = client.post("/agent", json={"tool": "get_weather", "argument": "Delhi"})
    assert r.status_code == 200
    assert "hazy" in r.json()["result"]


def test_agent_unknown_tool():
    r = client.post("/agent", json={"tool": "not_a_tool"})
    assert r.status_code == 400
