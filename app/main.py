"""
Minimal Agentic Tool-Calling Service
-------------------------------------
A FastAPI service that exposes an /agent endpoint. The agent can call
one of a few simple tools (calculator, get_time, get_weather_mock) to
answer a user's request. This is intentionally small so the FOCUS of
this project stays on Git/Docker/CI-CD, not on agent complexity.
"""

import math
import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = "1.0.0"

app = FastAPI(title="Agentic Capstone Service", version=APP_VERSION)


# ---------- Tools the agent can call ----------

def tool_calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression):
        raise ValueError("Expression contains disallowed characters")
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")
    return str(result)


def tool_get_time() -> str:
    """Return the current UTC time."""
    return datetime.now(timezone.utc).isoformat()


def tool_get_weather_mock(city: str) -> str:
    """Mocked weather tool (no external API call, deterministic for tests)."""
    fake_data = {
        "san francisco": "62F, foggy",
        "new york": "75F, sunny",
        "delhi": "34C, hazy",
        "banswara": "33C, partly cloudy",
    }
    return fake_data.get(city.lower(), f"No weather data for '{city}' (mock service)")


TOOLS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
    "get_weather": tool_get_weather_mock,
}


# ---------- API models ----------

class AgentRequest(BaseModel):
    tool: str
    argument: Optional[str] = None


class AgentResponse(BaseModel):
    tool: str
    result: str
    latency_ms: float


# ---------- Routes ----------

@app.get("/health")
def health():
    """Used by CD pipeline / container orchestrator for health checks."""
    return {"status": "ok", "time": tool_get_time()}


@app.get("/")
def root():
    return {"message": "Agentic Capstone Service is running", "tools": list(TOOLS.keys())}


@app.get("/version")
def version():
    return {"version": APP_VERSION}


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    if req.tool not in TOOLS:
        available = list(TOOLS.keys())
        detail = f"Unknown tool '{req.tool}'. Available: {available}"
        raise HTTPException(status_code=400, detail=detail)

    start = time.perf_counter()
    try:
        if req.tool == "get_time":
            result = tool_get_time()
        elif req.tool == "calculator":
            if not req.argument:
                raise HTTPException(status_code=400, detail="calculator requires 'argument'")
            result = tool_calculator(req.argument)
        elif req.tool == "get_weather":
            if not req.argument:
                detail = "get_weather requires 'argument' (city)"
                raise HTTPException(status_code=400, detail=detail)
            result = tool_get_weather_mock(req.argument)
        else:
            raise HTTPException(status_code=400, detail="Tool not implemented")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    return AgentResponse(tool=req.tool, result=result, latency_ms=round(latency_ms, 3))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
