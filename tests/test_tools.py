from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reasoning_chain.tools import ToolError, web_search


def grounded_response(*, include_sources=True):
    chunks = []
    if include_sources:
        chunks = [
            SimpleNamespace(
                web=SimpleNamespace(title="Primary source", uri="https://example.com/fact")
            ),
            SimpleNamespace(
                web=SimpleNamespace(title="Duplicate", uri="https://example.com/fact")
            ),
            SimpleNamespace(
                web=SimpleNamespace(title="Second source", uri="https://example.org/report")
            ),
        ]
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        web_search_queries=["current fact"],
        search_entry_point=SimpleNamespace(rendered_content="suggestions"),
    )
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )
    return SimpleNamespace(
        text="The grounded answer.",
        candidates=[SimpleNamespace(grounding_metadata=metadata)],
        usage_metadata=usage,
    )


def test_web_search_returns_grounded_answer_sources_and_telemetry():
    with patch("reasoning_chain.tools._get_search_client") as get_client:
        get_client.return_value.models.generate_content.return_value = grounded_response()
        output, api_calls = web_search("current fact")

    assert "The grounded answer." in output
    assert "[Primary source](https://example.com/fact)" in output
    assert output.count("https://example.com/fact") == 1
    assert "[Second source](https://example.org/report)" in output
    assert api_calls[0]["method"] == "GEMINI_SEARCH"
    assert api_calls[0]["response_payload"]["queries"] == ["current fact"]
    assert api_calls[0]["response_payload"]["usage"]["total_tokens"] == 30


def test_web_search_rejects_an_answer_without_public_sources():
    with patch("reasoning_chain.tools._get_search_client") as get_client:
        get_client.return_value.models.generate_content.return_value = grounded_response(
            include_sources=False
        )
        with pytest.raises(ToolError, match="no grounded answer") as exc_info:
            web_search("current fact")

    assert exc_info.value.api_calls[0]["status"] == 502


def test_web_search_records_provider_failure_without_leaking_a_key():
    with patch("reasoning_chain.tools._get_search_client") as get_client:
        get_client.return_value.models.generate_content.side_effect = RuntimeError(
            "provider unavailable"
        )
        with pytest.raises(ToolError, match="web search failed") as exc_info:
            web_search("current fact")

    assert exc_info.value.api_calls[0]["status"] == 500
    assert "provider unavailable" in exc_info.value.api_calls[0]["response_payload"]["error"]
