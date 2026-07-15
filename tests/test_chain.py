"""
Tests for the reasoning chain.

Production note: these mock the Anthropic calls entirely. You don't want
CI hitting a real LLM API on every push -- it's slow, costs money, and
non-deterministic output makes tests flaky. Mock the model, test that
*your* orchestration logic (retries, circuit breaker, repair loop)
behaves correctly regardless of what the model says.

Run with: pytest reasoning_chain/test_chain.py -v
"""

from unittest.mock import patch

import pytest

from reasoning_chain.chain import execute_plan, run_chain
from reasoning_chain.schemas import Plan, PlanStep, StepResult, VerifyResult


def test_execute_plan_all_succeed():
    plan = Plan(
        goal="what time is it",
        steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="user asked")],
    )
    results = execute_plan(plan)
    assert len(results) == 1
    assert results[0].success is True


def test_execute_plan_retries_then_succeeds():
    plan = Plan(
        goal="weather in Paris",
        steps=[
            PlanStep(
                step_id=1, tool="weather",
                tool_input={"city": "Paris"}, reason="check weather",
            ),
        ],
    )
    # Fail once, then succeed -- exercises the retry path without depending
    # on the real random failure rate.
    calls = {"n": 0}

    def flaky_weather(city):
        calls["n"] += 1
        if calls["n"] == 1:
            from reasoning_chain.tools import ToolError

            raise ToolError("simulated timeout")
        return f"{city}: clear, 20C"

    with patch.dict("reasoning_chain.chain.TOOL_REGISTRY", {"weather": flaky_weather}):
        results = execute_plan(plan)

    assert results[0].success is True
    assert results[0].attempt == 2


def test_execute_plan_circuit_breaker_disables_repeated_failures():
    plan = Plan(
        goal="two weather checks",
        steps=[
            PlanStep(step_id=1, tool="weather", tool_input={"city": "A"}, reason="r1"),
            PlanStep(step_id=2, tool="weather", tool_input={"city": "B"}, reason="r2"),
        ],
    )

    def always_fails(city):
        from reasoning_chain.tools import ToolError

        raise ToolError("service down")

    with patch.dict("reasoning_chain.chain.TOOL_REGISTRY", {"weather": always_fails}):
        results = execute_plan(plan)

    assert results[0].success is False
    # second step should be skipped by the circuit breaker, not retried again
    assert results[1].success is False
    assert "skipped" in results[1].error


def test_run_chain_stops_after_max_repair_rounds():
    """Even if the model keeps saying 'not satisfied', the loop must
    terminate -- this is the test that would catch an infinite-loop bug
    before it reaches production."""
    fake_plan = Plan(
        goal="impossible goal",
        steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="r")],
    )
    never_satisfied = VerifyResult(
        satisfied=False,
        missing=["something"],
        repair_steps=[PlanStep(step_id=99, tool="get_time", tool_input={}, reason="retry")],
        final_summary="still missing data",
    )

    with patch("reasoning_chain.chain.decompose_goal", return_value=fake_plan), patch(
        "reasoning_chain.chain.verify_and_repair", return_value=never_satisfied
    ):
        trace = run_chain("impossible goal")

    assert trace.repair_rounds == 1  # MAX_REPAIR_ROUNDS, not infinite
    assert trace.verify.satisfied is False


def test_run_chain_stops_immediately_when_satisfied():
    fake_plan = Plan(
        goal="what time is it",
        steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="r")],
    )
    satisfied = VerifyResult(satisfied=True, final_summary="done")

    with patch("reasoning_chain.chain.decompose_goal", return_value=fake_plan), patch(
        "reasoning_chain.chain.verify_and_repair", return_value=satisfied
    ):
        trace = run_chain("what time is it")

    assert trace.repair_rounds == 0
    assert trace.verify.satisfied is True


def test_execute_plan_resolves_step_references():
    from reasoning_chain.chain import _resolve_references
    from reasoning_chain.schemas import ToolName

    history = [
        StepResult(
            step_id=1,
            tool=ToolName.weather,
            tool_input={"city": "London"},
            output="London: Partly cloudy, 22.4°C",
            success=True,
            attempt=1,
            latency_ms=10.0,
            api_calls=[]
        ),
        StepResult(
            step_id=2,
            tool=ToolName.weather,
            tool_input={"city": "Jaipur"},
            output="Jaipur: Mist, 34.0°C",
            success=True,
            attempt=1,
            latency_ms=10.0,
            api_calls=[]
        )
    ]

    inp = {"expression": "[1] * 3"}
    res = _resolve_references(inp, history)
    assert res["expression"] == "22.4 * 3"

    inp2 = {"expression": "([1] + [2]) - 10"}
    res2 = _resolve_references(inp2, history)
    assert res2["expression"] == "(22.4 + 34.0) - 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
