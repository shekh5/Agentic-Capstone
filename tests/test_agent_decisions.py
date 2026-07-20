import pytest

from reasoning_chain.decisions import DecisionValidationError, parse_agent_decision
from reasoning_chain.schemas import ActionDecision, CorrectionType, FinalDecision, ToolName


def test_parses_typed_action_decision_and_validates_tool_input():
    decision = parse_agent_decision(
        '{"state":"action","reason":"Need weather.","tool":"weather",'
        '"tool_input":{"city":"Delhi"}}'
    )

    assert isinstance(decision, ActionDecision)
    assert decision.tool == ToolName.weather
    assert decision.tool_input == {"city": "Delhi"}


def test_normalizes_legacy_thought_response():
    decision = parse_agent_decision(
        '{"thought":"Need arithmetic.","tool":"calculator",'
        '"tool_input":{"expression":"2 + 2"}}'
    )

    assert isinstance(decision, ActionDecision)
    assert decision.state == "action"
    assert decision.reason == "Need arithmetic."


def test_parses_honest_unsatisfied_final_decision():
    decision = parse_agent_decision(
        '{"state":"final","satisfied":false,"final_summary":"Service unavailable."}'
    )

    assert isinstance(decision, FinalDecision)
    assert decision.satisfied is False


@pytest.mark.parametrize(
    ("raw", "correction_type"),
    [
        ("", CorrectionType.empty_response),
        ("not JSON", CorrectionType.json_parse_error),
        (
            '{"state":"action","reason":"Search.","tool":"search",'
            '"tool_input":{}}',
            CorrectionType.unknown_tool,
        ),
        (
            '{"state":"action","reason":"Weather.","tool":"weather",'
            '"tool_input":{}}',
            CorrectionType.invalid_tool_input,
        ),
        (
            '{"state":"action","reason":"Weather.","tool":"weather",'
            '"tool_input":{"city":"   "}}',
            CorrectionType.invalid_tool_input,
        ),
    ],
)
def test_classifies_recoverable_decision_errors(raw, correction_type):
    with pytest.raises(DecisionValidationError) as exc_info:
        parse_agent_decision(raw)

    assert exc_info.value.correction_type == correction_type


def test_get_time_input_receives_safe_default_timezone():
    decision = parse_agent_decision(
        '{"state":"action","reason":"Need current time.","tool":"get_time",'
        '"tool_input":{}}'
    )

    assert decision.tool_input == {"timezone_name": "UTC"}


def test_web_search_input_is_validated_and_trimmed():
    decision = parse_agent_decision(
        '{"state":"action","reason":"Need current sources.","tool":"web_search",'
        '"tool_input":{"query":"  latest Gemini release  "}}'
    )

    assert decision.tool == ToolName.web_search
    assert decision.tool_input == {"query": "latest Gemini release"}


def test_web_search_rejects_empty_query():
    with pytest.raises(DecisionValidationError) as exc_info:
        parse_agent_decision(
            '{"state":"action","reason":"Search.","tool":"web_search",'
            '"tool_input":{"query":" "}}'
        )

    assert exc_info.value.correction_type == CorrectionType.invalid_tool_input
