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

import json
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .safe_math import evaluate_arithmetic


def _failure_rate(name: str) -> float:
    try:
        value = float(os.environ.get(name, "0"))
    except ValueError:
        return 0.0
    return min(max(value, 0.0), 1.0)


WEATHER_FAILURE_RATE = _failure_rate("WEATHER_FAILURE_RATE")
CALCULATOR_BAD_INPUT_RATE = _failure_rate("CALCULATOR_BAD_INPUT_RATE")


class ToolError(Exception):
    pass


def calculator(expression: str) -> str:
    if CALCULATOR_BAD_INPUT_RATE and random.random() < CALCULATOR_BAD_INPUT_RATE:
        # Simulate the LLM having produced something eval can't parse,
        # e.g. "120 * remaining_budget" with an undefined name.
        raise ToolError(f"could not parse expression: {expression!r}")
    try:
        return evaluate_arithmetic(expression)
    except ValueError as exc:
        raise ToolError(f"evaluation failed: {exc}") from exc


def get_time(timezone_name: str = "UTC") -> str:
    try:
        requested_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ToolError(f"unknown timezone: {timezone_name!r}") from exc
    return datetime.now(timezone.utc).astimezone(requested_timezone).isoformat()


def weather(city: str) -> tuple[str, list[dict]]:
    api_key = os.environ.get("WEATHER_API_KEY")
    api_calls = []
    if api_key:
        try:
            safe_city = urllib.parse.quote(city)
            url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={safe_city}"
            api_start = time.perf_counter()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    status_code = response.getcode()
                    resp_body = response.read().decode()
                    api_latency = (time.perf_counter() - api_start) * 1000
                    
                    data = json.loads(resp_body)
                    loc_name = data["location"]["name"]
                    cond = data["current"]["condition"]["text"]
                    temp = data["current"]["temp_c"]
                    
                    api_calls.append({
                        "url": url.replace(api_key, "REDACTED_KEY"),
                        "method": "GET",
                        "status": status_code,
                        "latency_ms": round(api_latency, 2),
                        "response_payload": data
                    })
                    return f"{loc_name}: {cond}, {temp}°C", api_calls
            except Exception as e:
                api_latency = (time.perf_counter() - api_start) * 1000
                api_calls.append({
                    "url": url.replace(api_key, "REDACTED_KEY"),
                    "method": "GET",
                    "status": getattr(e, "code", 500),
                    "latency_ms": round(api_latency, 2),
                    "response_payload": {"error": str(e)}
                })
                te = ToolError(f"weather api call failed for {city!r}: {e}")
                te.api_calls = api_calls
                raise te
        except Exception as e:
            if isinstance(e, ToolError):
                raise e
            te = ToolError(f"weather api call failed for {city!r}: {e}")
            te.api_calls = api_calls
            raise te

    if WEATHER_FAILURE_RATE and random.random() < WEATHER_FAILURE_RATE:
        # Simulate a timeout to a third-party weather API.
        time.sleep(0.05)
        raise ToolError(f"weather service timed out for city={city!r}")
    # Mock response -- swap for a real API call in production.
    condition = "clear"
    temp_c = 20
    return f"{city}: {condition}, {temp_c}\u00b0C", api_calls


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_time": get_time,
    "weather": weather,
}
