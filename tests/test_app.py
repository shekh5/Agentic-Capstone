from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from reasoning_chain.schemas import ChainTrace, Plan, PlanStep, VerifyResult

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "tools" in r.json()


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"version": "1.0.0"}


def test_chat_ui():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Agentic Capstone" in r.text


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


def test_chain_plan_route_is_mounted():
    fake_plan = Plan(
        goal="what time is it",
        steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="user asked")],
    )

    with patch("reasoning_chain.router.decompose_goal", return_value=fake_plan):
        r = client.post("/chain/plan", params={"goal": "what time is it"})

    assert r.status_code == 200
    assert r.json()["goal"] == "what time is it"
    assert r.json()["steps"][0]["tool"] == "get_time"


def test_chain_run_route_is_mounted():
    fake_trace = ChainTrace(
        request_id="req-1",
        goal="what time is it",
        plan=Plan(
            goal="what time is it",
            steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="user asked")],
        ),
        results=[],
        verify=VerifyResult(satisfied=True, final_summary="done"),
        repair_rounds=0,
    )

    with patch("reasoning_chain.router.run_chain", return_value=fake_trace):
        r = client.post("/chain/run", params={"goal": "what time is it"})

    assert r.status_code == 200
    assert r.json()["request_id"] == "req-1"
