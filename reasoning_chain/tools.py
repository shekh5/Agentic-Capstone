"""
Your existing calculator / get_time / weather tools, wrapped with an
injectable failure mode so you can *prove* the chain handles failure
gracefully instead of hoping it does.

Production note:
Real production agents get this "for free" because real APIs fail on
their own -- timeouts, rate limits, bad data. In dev, nothing ever fails,
so you never actually exercise your retry/repair code paths until it's
3am and something breaks in prod. Injecting failure locally is how you
pull that 3am bug into your test suite instead.
"""

import random
import time
from datetime import datetime, timezone

# Toggle / tune via env vars in real deployment; hardcoded here for clarity.
WEATHER_FAILURE_RATE = 0.35
CALCULATOR_BAD_INPUT_RATE = 0.15


class ToolError(Exception):
    pass


def calculator(expression: str) -> str:
    if random.random() < CALCULATOR_BAD_INPUT_RATE:
        # Simulate the LLM having produced something eval can't parse,
        # e.g. "120 * remaining_budget" with an undefined name.
        raise ToolError(f"could not parse expression: {expression!r}")
    try:
        # NOTE: eval() here is for local learning/demo purposes only.
        # A real deployment should use a safe expression parser
        # (e.g. `asteval` or a small hand-written grammar) -- never eval()
        # untrusted/LLM-generated strings in production.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        raise ToolError(f"evaluation failed: {e}")


def get_time(timezone_name: str = "UTC") -> str:
    return datetime.now(timezone.utc).isoformat()


def weather(city: str) -> str:
    if random.random() < WEATHER_FAILURE_RATE:
        # Simulate a timeout to a third-party weather API.
        time.sleep(0.05)
        raise ToolError(f"weather service timed out for city={city!r}")
    # Mock response -- swap for a real API call in production.
    condition = random.choice(["clear", "rainy", "cloudy", "windy"])
    temp_c = random.randint(10, 30)
    return f"{city}: {condition}, {temp_c}\u00b0C"


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_time": get_time,
    "weather": weather,
}
