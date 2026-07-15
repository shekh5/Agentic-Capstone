"""
The orchestration layer: decompose -> execute -> verify -> (repair) -> summarize.

Production note:
This whole file is the "deterministic agent" pattern you already know from
CI/CD, applied to LLM calls:
  - decompose_goal()  ~ the pipeline's "plan" stage (like a CI config parse)
  - execute_plan()    ~ the job runner, with retries + a circuit breaker
  - verify_and_repair ~ a post-build check that can trigger one more
                         limited "re-run" instead of looping forever
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

from .schemas import ChainTrace, Plan, StepResult, VerifyResult, ModelCall
from .tools import TOOL_REGISTRY, ToolError

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6
MAX_REPAIR_ROUNDS = 1
MAX_TOOL_RETRIES = 1

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
                "google-genai is not installed; install dependencies or patch decompose_goal/verify_and_repair "
                "in tests"
            )
        _client = genai.Client()
    return _client


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
    system = (
        "You break a user's goal into a short sequence of tool calls. "
        "Available tools:\n"
        "- calculator(expression: str) -> tool_input MUST be a dictionary like {\"expression\": \"2 + 2\"}\n"
        "- get_time(timezone_name: str) -> tool_input MUST be a dictionary like {\"timezone_name\": \"UTC\"}\n"
        "- weather(city: str) -> tool_input MUST be a dictionary like {\"city\": \"Delhi\"}\n"
        "If a step depends on a value returned by a previous step (e.g. using the temperature from step 1 in a calculation), represent that value using step reference format like '[1]' where 1 is the step_id. For example: '([1] * 3) - 5'. Do not write text variable names like London_temp.\n"
        f"Use at most {MAX_STEPS} steps. "
        "Respond with ONLY a JSON object matching this shape, no prose, "
        "no markdown fences: "
        '{"goal": str, "steps": [{"step_id": int, "tool": str, '
        '"tool_input": dict, "reason": str}]}'
    )
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=goal,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        model_calls.append(
            ModelCall(
                stage="decompose",
                system_prompt=system,
                user_prompt=goal,
                raw_response=raw,
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


def execute_plan(plan: Plan) -> list[StepResult]:
    """Stage 2: run each step against real tools with retry + circuit
    breaker. A tool that fails twice in a row gets skipped for the rest
    of the run instead of retried forever."""
    results: list[StepResult] = []
    disabled_tools: set[str] = set()

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

        resolved_input = _resolve_references(step.tool_input, results)
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
    system = (
        "You check whether tool results satisfy the user's goal. "
        "If something failed or is missing, propose at most 2 repair "
        "steps using the same tools. Available tools:\n"
        "- calculator(expression: str) -> tool_input MUST be a dictionary like {\"expression\": \"2 + 2\"}\n"
        "- get_time(timezone_name: str) -> tool_input MUST be a dictionary like {\"timezone_name\": \"UTC\"}\n"
        "- weather(city: str) -> tool_input MUST be a dictionary like {\"city\": \"Delhi\"}\n"
        "If a step depends on a value returned by a previous step (e.g. using the temperature from step 1 in a calculation), represent that value using step reference format like '[1]' where 1 is the step_id. For example: '([1] * 3) - 5'. Do not write text variable names like London_temp.\n"
        "If a repair isn't possible, say so "
        "plainly in final_summary instead of guessing. "
        "Respond with ONLY JSON: {\"satisfied\": bool, \"missing\": "
        "[str], \"repair_steps\": [{\"step_id\": int, \"tool\": str, "
        "\"tool_input\": dict, \"reason\": str}], \"final_summary\": str}"
    )
    payload = {
        "goal": goal,
        "results": [r.model_dump() for r in results],
    }
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=json.dumps(payload),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        model_calls.append(
            ModelCall(
                stage="verify",
                system_prompt=system,
                user_prompt=json.dumps(payload),
                raw_response=raw,
            )
        )
    data = _extract_json(raw)
    return VerifyResult.model_validate(data)


def run_chain(goal: str) -> ChainTrace:
    """The full orchestrator. This is the function your FastAPI route calls."""
    start_time = datetime.now(timezone.utc).isoformat()
    chain_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    model_calls = []

    plan = decompose_goal(goal, model_calls)
    results = execute_plan(plan)
    verify = verify_and_repair(goal, plan, results, model_calls)

    repair_rounds = 0
    while not verify.satisfied and verify.repair_steps and repair_rounds < MAX_REPAIR_ROUNDS:
        repair_plan = Plan(goal=goal, steps=verify.repair_steps)
        repair_results = execute_plan(repair_plan)
        results = results + repair_results
        verify = verify_and_repair(goal, plan, results, model_calls)
        repair_rounds += 1

    total_latency_ms = (time.perf_counter() - chain_start) * 1000
    end_time = datetime.now(timezone.utc).isoformat()

    return ChainTrace(
        request_id=request_id,
        goal=goal,
        plan=plan,
        results=results,
        verify=verify,
        repair_rounds=repair_rounds,
        start_time=start_time,
        end_time=end_time,
        total_latency_ms=round(total_latency_ms, 2),
        llm_calls=model_calls,
    )
