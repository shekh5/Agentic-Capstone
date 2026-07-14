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
import uuid

from .schemas import ChainTrace, Plan, StepResult, VerifyResult
from .tools import TOOL_REGISTRY, ToolError

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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


def decompose_goal(goal: str) -> Plan:
    """Stage 1: ask the model to break the goal into tool calls, returned
    as structured JSON only -- not prose."""
    system = (
        "You break a user's goal into a short sequence of tool calls. "
        "Available tools: calculator(expression: str), "
        "get_time(timezone_name: str), weather(city: str). "
        f"Use at most {MAX_STEPS} steps. "
        "Respond with ONLY a JSON object matching this shape, no prose, "
        "no markdown fences: "
        '{"goal": str, "steps": [{"step_id": int, "tool": str, '
        '"tool_input": object, "reason": str}]}'
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
    data = _extract_json(raw)
    return Plan.model_validate(data)


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

        fn = TOOL_REGISTRY[step.tool]
        last_error = None
        for attempt in range(1, MAX_TOOL_RETRIES + 2):
            try:
                output = fn(**step.tool_input)
                results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool=step.tool,
                        tool_input=step.tool_input,
                        output=output,
                        success=True,
                        attempt=attempt,
                    )
                )
                break
            except ToolError as e:
                last_error = str(e)
            except TypeError as e:
                last_error = f"bad tool_input: {e}"
                break
        else:
            results.append(
                StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    tool_input=step.tool_input,
                    success=False,
                    error=last_error,
                    attempt=MAX_TOOL_RETRIES + 1,
                )
            )
            disabled_tools.add(step.tool)

    return results


def verify_and_repair(goal: str, plan: Plan, results: list[StepResult]) -> VerifyResult:
    """Stage 3: ask the model to check tool outputs against the goal. It
    can either declare victory, ask for a small number of repair steps,
    or -- critically -- admit what it couldn't determine rather than
    hallucinating a confident-sounding answer."""
    system = (
        "You check whether tool results satisfy the user's goal. "
        "If something failed or is missing, propose at most 2 repair "
        "steps using the same tools. If a repair isn't possible, say so "
        "plainly in final_summary instead of guessing. "
        "Respond with ONLY JSON: {\"satisfied\": bool, \"missing\": "
        "[str], \"repair_steps\": [{\"step_id\": int, \"tool\": str, "
        "\"tool_input\": object, \"reason\": str}], \"final_summary\": str}"
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
    data = _extract_json(raw)
    return VerifyResult.model_validate(data)


def run_chain(goal: str) -> ChainTrace:
    """The full orchestrator. This is the function your FastAPI route calls."""
    request_id = str(uuid.uuid4())
    plan = decompose_goal(goal)
    results = execute_plan(plan)
    verify = verify_and_repair(goal, plan, results)

    repair_rounds = 0
    while not verify.satisfied and verify.repair_steps and repair_rounds < MAX_REPAIR_ROUNDS:
        repair_plan = Plan(goal=goal, steps=verify.repair_steps)
        repair_results = execute_plan(repair_plan)
        results = results + repair_results
        verify = verify_and_repair(goal, plan, results)
        repair_rounds += 1

    return ChainTrace(
        request_id=request_id,
        goal=goal,
        plan=plan,
        results=results,
        verify=verify,
        repair_rounds=repair_rounds,
    )
