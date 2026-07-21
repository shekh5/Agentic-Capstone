from types import SimpleNamespace
from unittest.mock import patch

from reasoning_chain.chain import run_chain
from reasoning_chain.context import ContextBundle
from reasoning_chain.tools import TOOL_REGISTRY, calculator


def response(text):
    return SimpleNamespace(text=text, usage_metadata=None)


def test_invalid_json_is_corrected_without_consuming_tool_step():
    responses = [
        response("I should use a calculator."),
        response(
            '{"state":"action","reason":"Arithmetic is required.",'
            '"tool":"calculator","tool_input":{"expression":"2 + 2"}}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            '"final_summary":"The result is 4."}'
        ),
    ]

    with patch("reasoning_chain.chain._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("calculate two plus two")

    assert len(trace.results) == 1
    assert trace.results[0].step_id == 1
    assert trace.results[0].output == "4"
    assert [call.stage for call in trace.llm_calls] == [
        "step_1",
        "step_1_correction_1",
        "step_2",
    ]
    assert trace.total_corrections == 1
    assert trace.corrections[0].correction_type == "json_parse_error"
    assert trace.corrections[0].successful is True
    assert "<correction_request" in trace.llm_calls[1].user_prompt
    assert "I should use a calculator" not in trace.llm_calls[1].system_prompt


def test_invalid_output_exhaustion_returns_controlled_trace():
    responses = [response("bad output"), response("still bad"), response("")]

    with patch("reasoning_chain.chain._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("unrecoverable decision")

    assert trace.verify.satisfied is False
    assert "could not produce a valid decision" in trace.verify.final_summary
    assert trace.verify.missing == ["Valid agent decision"]
    assert len(trace.llm_calls) == 3
    assert trace.total_corrections == 2
    assert all(record.successful is False for record in trace.corrections)
    assert trace.corrections[-1].result_error == "The model returned an empty response."


def test_missing_tool_input_is_corrected_before_execution():
    responses = [
        response(
            '{"state":"action","reason":"Need weather.",'
            '"tool":"weather","tool_input":{}}'
        ),
        response(
            '{"state":"action","reason":"A city is required.",'
            '"tool":"weather","tool_input":{"city":"Delhi"}}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            '"final_summary":"Delhi is clear."}'
        ),
    ]

    with (
        patch("reasoning_chain.chain._get_client") as mock_client,
        patch.dict(
            "reasoning_chain.chain.TOOL_REGISTRY",
            {"weather": lambda city: f"{city}: clear"},
        ),
    ):
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("weather in Delhi")

    assert len(trace.results) == 1
    assert trace.results[0].tool_input == {"city": "Delhi"}
    assert trace.corrections[0].correction_type == "invalid_tool_input"
    assert trace.corrections[0].successful is True


def test_all_failed_tools_force_correction_of_unsupported_success():
    responses = [
        response(
            '{"state":"action","reason":"Try weather.",'
            '"tool":"weather","tool_input":{"city":"Delhi"}}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            '"final_summary":"Delhi is sunny."}'
        ),
        response(
            '{"state":"final","satisfied":false,'
            '"final_summary":"Weather is unavailable."}'
        ),
    ]

    def unavailable(city):
        from reasoning_chain.tools import ToolError

        raise ToolError(f"weather unavailable for {city}")

    with (
        patch("reasoning_chain.chain._get_client") as mock_client,
        patch.dict("reasoning_chain.chain.TOOL_REGISTRY", {"weather": unavailable}),
    ):
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("weather in Delhi")

    assert trace.verify.satisfied is False
    assert trace.verify.final_summary == "Weather is unavailable."
    assert trace.corrections[0].correction_type == "unsupported_final_answer"
    assert trace.corrections[0].successful is True


def test_runtime_agent_has_no_code_editing_tools():
    assert set(TOOL_REGISTRY) == {"calculator", "get_time", "weather", "web_search"}
    assert TOOL_REGISTRY["calculator"] is calculator
    assert not {"shell", "write_file", "edit_file", "apply_patch"} & set(TOOL_REGISTRY)


def test_web_answer_without_returned_source_is_corrected():
    source_url = "https://example.com/current-fact"
    responses = [
        response(
            '{"state":"action","reason":"Current information is required.",'
            '"tool":"web_search","tool_input":{"query":"latest fact"}}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            '"final_summary":"The current fact is 42."}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            f'"final_summary":"The current fact is 42. Source: {source_url}"}}'
        ),
    ]

    def grounded_search(query):
        return (
            f"The current fact is 42.\n\nSources:\n[1] [Example]({source_url})",
            [],
        )

    with (
        patch("reasoning_chain.chain._get_client") as mock_client,
        patch.dict(
            "reasoning_chain.chain.TOOL_REGISTRY",
            {"web_search": grounded_search},
        ),
    ):
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("What is the latest fact?")

    assert trace.verify.satisfied is True
    assert source_url in trace.verify.final_summary
    assert trace.corrections[0].correction_type == "missing_citations"
    assert trace.corrections[0].successful is True


def test_pdf_answer_without_page_citation_is_corrected():
    citation = "[report.pdf, page 3]"
    conversation = ContextBundle(
        document_context=(
            '<document_context trust="untrusted">'
            f'<passage citation="{citation}">Revenue grew 18 percent.</passage>'
            "</document_context>"
        ),
        document_citations=[citation],
        document_ids=["doc-1"],
        document_chunks=1,
    )
    responses = [
        response(
            '{"state":"final","satisfied":true,'
            '"final_summary":"Revenue grew 18 percent."}'
        ),
        response(
            '{"state":"final","satisfied":true,'
            f'"final_summary":"Revenue grew 18 percent {citation}."}}'
        ),
    ]

    with patch("reasoning_chain.chain._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = responses
        trace = run_chain("How much did revenue grow?", conversation=conversation)

    assert trace.verify.satisfied is True
    assert citation in trace.verify.final_summary
    assert trace.document_ids == ["doc-1"]
    assert trace.corrections[0].correction_type == "missing_citations"
