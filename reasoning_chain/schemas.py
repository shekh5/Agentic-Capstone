"""
Structured data contracts for the reasoning chain.

Production note:
Everything passed between stages (decompose -> execute -> verify) is a
validated Pydantic model, never raw text. This is the single biggest
difference between a "prompt chain demo" and something you'd trust in
production: if the LLM's JSON doesn't match the schema, Pydantic raises
immediately at the boundary instead of a malformed value silently
propagating three steps downstream where it's much harder to debug.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    calculator = "calculator"
    get_time = "get_time"
    weather = "weather"


class PlanStep(BaseModel):
    step_id: int
    tool: ToolName
    tool_input: dict = Field(default_factory=dict)
    reason: str = Field(..., description="Why this step is needed, for tracing/debugging")


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]


class ModelCall(BaseModel):
    stage: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StepResult(BaseModel):
    step_id: int
    tool: ToolName
    tool_input: dict
    output: Optional[str] = None
    success: bool
    error: Optional[str] = None
    attempt: int = 1
    latency_ms: float = 0.0
    api_calls: list[dict] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class VerifyResult(BaseModel):
    satisfied: bool
    missing: list[str] = Field(default_factory=list)
    repair_steps: list[PlanStep] = Field(default_factory=list)
    final_summary: str


class ChainTrace(BaseModel):
    """The full record of one run. This is what you'd log to Redis / your
    observability tool -- one document per request_id, every stage included."""

    request_id: str
    goal: str
    plan: Plan
    results: list[StepResult]
    verify: VerifyResult
    repair_rounds: int
    start_time: str = ""
    end_time: str = ""
    total_latency_ms: float = 0.0
    llm_calls: list[ModelCall] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    session_id: Optional[str] = None


class SessionMessage(BaseModel):
    sender: str
    text: str
    timestamp: str
    trace: Optional[dict] = None
