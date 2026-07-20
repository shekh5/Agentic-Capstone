"""
Reasoning orchestration with a bounded ReAct loop and reusable planning helpers.

Production note:
The active run_chain() flow asks the model for one action at a time, executes
that action with bounded retries, and stops after eight actions. The standalone
decompose_goal() and verify_and_repair() helpers support plan inspection and
experimentation but are not stages in run_chain().
No step trusts the previous step's output blindly -- every boundary is a
Pydantic model (see schemas.py), and every stage is logged with a
request_id so a failed run can be replayed/debugged like a CI run.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .context import ContextBundle, ContextMessage, build_budgeted_contents, estimate_tokens
from .prompts import (
    DECOMPOSE_PROMPT_VERSION,
    REACT_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    VERIFY_PROMPT_VERSION,
    build_decompose_system_prompt,
    build_goal_request,
    build_react_request,
    build_react_system_prompt,
    build_summary_request,
    build_verify_request,
    build_verify_system_prompt,
)
from .schemas import ChainTrace, ModelCall, Plan, PlanStep, StepResult, VerifyResult
from .tools import TOOL_REGISTRY, ToolError

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6
MAX_REPAIR_ROUNDS = 1
MAX_TOOL_RETRIES = 1
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0


def _temperature_from_env(name: str, default: float) -> float:
    """Read a temperature without allowing bad deployment config to break startup."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(MIN_TEMPERATURE, min(value, MAX_TEMPERATURE))


REACT_TEMPERATURE = _temperature_from_env("REACT_TEMPERATURE", 0.1)
SUMMARY_TEMPERATURE = _temperature_from_env("SUMMARY_TEMPERATURE", 0.2)

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore

_client = None


def _get_client():
    global _client
    if _client is None:
        if genai is None:
            raise RuntimeError(
                "google-genai is not installed; install dependencies or patch "
                "decompose_goal/verify_and_repair in tests"
            )
        _client = genai.Client()
    return _client


def _count_tokens(contents: list[dict], system_instruction: str) -> int:
    """Use Gemini's tokenizer when available and a conservative local fallback otherwise."""
    try:
        response = _get_client().models.count_tokens(
            model=MODEL,
            contents=contents,
            config=types.CountTokensConfig(system_instruction=system_instruction),
        )
        count = getattr(response, "total_tokens", None)
        if isinstance(count, int) and count > 0:
            return count
    except Exception:
        pass
    return estimate_tokens(contents, system_instruction)


def summarize_context(existing_summary: str, messages: list[ContextMessage]) -> str:
    """Merge older conversation turns into compact, instruction-safe memory."""
    payload = build_summary_request(
        existing_summary,
        [{"role": message.role, "text": message.text} for message in messages],
    )
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            max_output_tokens=600,
            temperature=SUMMARY_TEMPERATURE,
        ),
    )
    return (response.text or "").strip()


def _extract_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences even when told not
    to. Production code defends against that instead of assuming a clean
    parse -- this is the same instinct as validating CI config before
    running it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def decompose_goal(goal: str, model_calls: list = None) -> Plan:
    """Stage 1: ask the model to break the goal into tool calls, returned
    as structured JSON only -- not prose."""
    system = build_decompose_system_prompt(MAX_STEPS)
    user_prompt = build_goal_request(goal)
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
            temperature=REACT_TEMPERATURE,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        p_tok = 0
        c_tok = 0
        t_tok = 0
        if resp.usage_metadata:
            p_tok = resp.usage_metadata.prompt_token_count or 0
            c_tok = resp.usage_metadata.candidates_token_count or 0
            t_tok = resp.usage_metadata.total_token_count or 0
        model_calls.append(
            ModelCall(
                stage="decompose",
                system_prompt=system,
                user_prompt=user_prompt,
                raw_response=raw,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                temperature=REACT_TEMPERATURE,
                prompt_version=DECOMPOSE_PROMPT_VERSION,
            )
        )
    data = _extract_json(raw)
    return Plan.model_validate(data)


def _resolve_references(tool_input: dict, results_history: list[StepResult]) -> dict:
    """
    Scans values in tool_input for step reference patterns like [id] (e.g. '[1]') 
    and replaces them with the parsed numeric output of that step from results_history.
    """
    resolved = {}
    for k, v in tool_input.items():
        if isinstance(v, str):
            def replacer(match):
                ref_id = int(match.group(1))
                for res in results_history:
                    if res.step_id == ref_id and res.success and res.output:
                        # Extract all numbers from the target step output
                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", res.output)
                        if nums:
                            return nums[-1]  # Take the last parsed number
                return match.group(0)
            
            resolved[k] = re.sub(r"\[(\d+)\]", replacer, v)
        else:
            resolved[k] = v
    return resolved


def execute_plan(
    plan: Plan,
    results_history: Optional[list[StepResult]] = None,
    disabled_tools: Optional[set[str]] = None,
) -> list[StepResult]:
    """Stage 2: run each step against real tools with retry + circuit
    breaker. A tool that fails twice in a row gets skipped for the rest
    of the run instead of retried forever."""
    results: list[StepResult] = []
    prior_results = results_history or []
    disabled_tools = disabled_tools if disabled_tools is not None else set()

    for step in plan.steps[:MAX_STEPS]:
        if step.tool in disabled_tools:
            results.append(
                StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    tool_input=step.tool_input,
                    success=False,
                    error="skipped: tool disabled after repeated failures",
                )
            )
            continue

        resolved_input = _resolve_references(step.tool_input, prior_results + results)
        fn = TOOL_REGISTRY[step.tool]
        last_error = None
        last_api_calls = []
        step_latency = 0.0

        for attempt in range(1, MAX_TOOL_RETRIES + 2):
            step_start = time.perf_counter()
            try:
                res = fn(**resolved_input)
                step_latency = (time.perf_counter() - step_start) * 1000
                if isinstance(res, tuple):
                    output, api_calls = res
                else:
                    output, api_calls = res, []

                results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool=step.tool,
                        tool_input=resolved_input,
                        output=output,
                        success=True,
                        attempt=attempt,
                        latency_ms=round(step_latency, 2),
                        api_calls=api_calls,
                    )
                )
                break
            except ToolError as e:
                step_latency = (time.perf_counter() - step_start) * 1000
                last_error = str(e)
                last_api_calls = getattr(e, "api_calls", [])
            except TypeError as e:
                step_latency = (time.perf_counter() - step_start) * 1000
                last_error = f"bad tool_input: {e}"
                last_api_calls = []
                results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool=step.tool,
                        tool_input=resolved_input,
                        success=False,
                        error=last_error,
                        attempt=attempt,
                        latency_ms=round(step_latency, 2),
                    )
                )
                break
        else:
            results.append(
                StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    tool_input=resolved_input,
                    success=False,
                    error=last_error,
                    attempt=MAX_TOOL_RETRIES + 1,
                    latency_ms=round(step_latency, 2),
                    api_calls=last_api_calls,
                )
            )
            disabled_tools.add(step.tool)

    return results


def verify_and_repair(
    goal: str, plan: Plan, results: list[StepResult], model_calls: list = None
) -> VerifyResult:
    """Stage 3: ask the model to check tool outputs against the goal. It
    can either declare victory, ask for a small number of repair steps,
    or -- critically -- admit what it couldn't determine rather than
    hallucinating a confident-sounding answer."""
    system = build_verify_system_prompt()
    payload = build_verify_request(goal, [r.model_dump() for r in results])
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
            temperature=REACT_TEMPERATURE,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        model_calls.append(
            ModelCall(
                stage="verify",
                system_prompt=system,
                user_prompt=payload,
                raw_response=raw,
                temperature=REACT_TEMPERATURE,
                prompt_version=VERIFY_PROMPT_VERSION,
            )
        )
    data = _extract_json(raw)
    return VerifyResult.model_validate(data)


def run_chain(
    goal: str,
    conversation: Optional[ContextBundle] = None,
    temperature: Optional[float] = None,
) -> ChainTrace:
    """The full orchestrator using a step-by-step ReAct loop."""
    effective_temperature = REACT_TEMPERATURE if temperature is None else temperature
    effective_temperature = max(MIN_TEMPERATURE, min(effective_temperature, MAX_TEMPERATURE))
    start_time = datetime.now(timezone.utc).isoformat()
    chain_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    model_calls = []
    results = []
    plan_steps = []
    disabled_tools: set[str] = set()
    
    satisfied = False
    final_summary = ""
    step_count = 0
    max_steps = 8

    system = build_react_system_prompt()

    while step_count < max_steps:
        # Build history payload for this step
        steps_taken = [
            {
                "step_id": r.step_id,
                "tool": r.tool,
                "tool_input": r.tool_input,
                "output": r.output,
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ]
        current_prompt = build_react_request(goal, steps_taken)
        contents = build_budgeted_contents(
            conversation,
            current_prompt,
            system,
            token_counter=_count_tokens,
        )
        resp = _get_client().models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1000,
                temperature=effective_temperature,
            ),
        )
        raw = resp.text
        
        p_tok = 0
        c_tok = 0
        t_tok = 0
        if resp.usage_metadata:
            p_tok = resp.usage_metadata.prompt_token_count or 0
            c_tok = resp.usage_metadata.candidates_token_count or 0
            t_tok = resp.usage_metadata.total_token_count or 0

        model_calls.append(
            ModelCall(
                stage=f"step_{step_count + 1}",
                system_prompt=system,
                user_prompt=current_prompt,
                raw_response=raw,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                temperature=effective_temperature,
                prompt_version=REACT_PROMPT_VERSION,
            )
        )
        
        data = _extract_json(raw)
        
        # Check if the model has finished
        if data.get("satisfied") is True:
            satisfied = True
            final_summary = data.get("final_summary", "Goal satisfied.")
            break
            
        # Otherwise, parse the next step proposal
        tool = data.get("tool")
        tool_input = data.get("tool_input", {})
        reason = data.get("reason", data.get("thought", ""))
        
        if not tool:
            satisfied = False
            final_summary = "Agent failed to propose a tool action."
            break
            
        step_id = step_count + 1
        
        plan_step = PlanStep(
            step_id=step_id,
            tool=tool,
            tool_input=tool_input,
            reason=reason,
        )
        plan_steps.append(plan_step)
        
        # Execute this single step using execute_plan
        single_plan = Plan(goal=goal, steps=[plan_step])
        step_results = execute_plan(
            single_plan,
            results_history=results,
            disabled_tools=disabled_tools,
        )
        
        if step_results:
            res = step_results[0]
            res.prompt_tokens = p_tok
            res.completion_tokens = c_tok
            res.total_tokens = t_tok
            results.append(res)
            
        step_count += 1
    else:
        satisfied = False
        final_summary = f"Stopped after reaching maximum of {max_steps} steps."

    total_prompt_tokens = sum(c.prompt_tokens for c in model_calls)
    total_completion_tokens = sum(c.completion_tokens for c in model_calls)
    total_tokens = sum(c.total_tokens for c in model_calls)

    total_latency_ms = (time.perf_counter() - chain_start) * 1000
    end_time = datetime.now(timezone.utc).isoformat()

    return ChainTrace(
        request_id=request_id,
        goal=goal,
        plan=Plan(goal=goal, steps=plan_steps),
        results=results,
        verify=VerifyResult(
            satisfied=satisfied,
            missing=[] if satisfied else ["Goal not fully met or stopped prematurely"],
            repair_steps=[],
            final_summary=final_summary
        ),
        repair_rounds=0,
        start_time=start_time,
        end_time=end_time,
        total_latency_ms=round(total_latency_ms, 2),
        llm_calls=model_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        temperature=effective_temperature,
    )
