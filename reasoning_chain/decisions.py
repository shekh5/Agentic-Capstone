"""Strict parsing and validation for model-produced agent decisions."""

import json
from typing import Union

from pydantic import TypeAdapter, ValidationError

from .schemas import (
    ActionDecision,
    CalculatorInput,
    CorrectionType,
    FinalDecision,
    GetTimeInput,
    ToolName,
    WeatherInput,
    WebSearchInput,
)

AgentDecision = Union[ActionDecision, FinalDecision]
_DECISION_ADAPTER = TypeAdapter(AgentDecision)
_TOOL_INPUT_MODELS = {
    ToolName.calculator: CalculatorInput,
    ToolName.get_time: GetTimeInput,
    ToolName.weather: WeatherInput,
    ToolName.web_search: WebSearchInput,
}


class DecisionValidationError(ValueError):
    def __init__(self, correction_type: CorrectionType, message: str):
        super().__init__(message)
        self.correction_type = correction_type
        self.message = message


def extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise DecisionValidationError(
            CorrectionType.empty_response,
            "The model returned an empty response.",
        )
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise DecisionValidationError(
            CorrectionType.json_parse_error,
            "No JSON object was found in the model response.",
        )
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise DecisionValidationError(
            CorrectionType.json_parse_error,
            f"The model response contained invalid JSON: {exc.msg}.",
        ) from exc
    if not isinstance(data, dict):
        raise DecisionValidationError(
            CorrectionType.schema_validation_error,
            "The model decision must be a JSON object.",
        )
    return data


def _normalize_legacy_decision(data: dict) -> dict:
    normalized = dict(data)
    if "state" not in normalized:
        if "tool" in normalized:
            normalized["state"] = "action"
        elif "satisfied" in normalized:
            normalized["state"] = "final"
    if "thought" in normalized:
        normalized.setdefault("reason", normalized["thought"])
        normalized.pop("thought")
    return normalized


def parse_agent_decision(raw_response: str) -> AgentDecision:
    data = _normalize_legacy_decision(extract_json(raw_response))
    if data.get("state") == "action":
        allowed_tools = {tool.value for tool in ToolName}
        if data.get("tool") not in allowed_tools:
            raise DecisionValidationError(
                CorrectionType.unknown_tool,
                f"Tool {data.get('tool')!r} is not registered. Allowed tools: "
                f"{', '.join(sorted(allowed_tools))}.",
            )
    try:
        decision = _DECISION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise DecisionValidationError(
            CorrectionType.schema_validation_error,
            _concise_validation_error(exc),
        ) from exc
    if isinstance(decision, ActionDecision):
        return validate_tool_input(decision)
    return decision


def validate_tool_input(decision: ActionDecision) -> ActionDecision:
    input_model = _TOOL_INPUT_MODELS[decision.tool]
    try:
        validated = input_model.model_validate(decision.tool_input)
    except ValidationError as exc:
        raise DecisionValidationError(
            CorrectionType.invalid_tool_input,
            f"Invalid input for {decision.tool.value}: {_concise_validation_error(exc)}",
        ) from exc
    return decision.model_copy(update={"tool_input": validated.model_dump()})


def action_fingerprint(decision: ActionDecision) -> str:
    return json.dumps(
        {"tool": decision.tool.value, "tool_input": decision.tool_input},
        sort_keys=True,
        separators=(",", ":"),
    )


def _concise_validation_error(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)[:1_000]
