"""
FastAPI endpoints for the reasoning chain.

Wire into your existing app with:

    from reasoning_chain.router import router as chain_router
    app.include_router(chain_router, prefix="/chain")

Production note on the Redis logging below:
Every run gets a request_id and its full ChainTrace is stored under that
key. This is the minimum viable version of what LangSmith / Helicone give
you in Phase 5 -- a way to pull up "what did the agent actually do for
this specific request" after the fact, instead of only having whatever
made it into stdout logs.
"""

import json
import logging
import os

from fastapi import APIRouter, HTTPException

from .chain import decompose_goal, run_chain
from .schemas import ChainTrace, Plan

logger = logging.getLogger("reasoning_chain")
router = APIRouter()

try:
    import redis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _redis = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
    _redis.ping()
except Exception as e:
    _redis = None
    logger.warning(f"Redis unavailable ({e}) -- chain traces will not be persisted")


def _log_trace(trace: ChainTrace) -> None:
    if _redis is None:
        return
    try:
        # Save trace content
        _redis.set(f"chain_trace:{trace.request_id}", trace.model_dump_json(), ex=60 * 60 * 24)
        # Add request_id to traces list for history visualization
        _redis.lpush("chain_traces_list", trace.request_id)
        # Keep only the last 100 trace IDs
        _redis.ltrim("chain_traces_list", 0, 99)
    except Exception as e:
        logger.warning(f"failed to persist trace {trace.request_id}: {e}")


@router.post("/plan", response_model=Plan)
def plan_only(goal: str):
    """Returns the decomposition only -- no tools executed. Use this to
    sanity-check the model's reasoning before wiring up execution."""
    try:
        return decompose_goal(goal)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"planning failed: {e}") from e


@router.post("/run", response_model=ChainTrace)
def run(goal: str):
    """Full plan -> execute -> verify -> repair loop. Returns the entire
    trace so the caller (or you, debugging) can see every step."""
    try:
        trace = run_chain(goal)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chain failed: {e}") from e
    _log_trace(trace)
    return trace


@router.get("/trace/{request_id}")
def get_trace(request_id: str):
    if _redis is None:
        raise HTTPException(status_code=503, detail="trace storage not configured")
    raw = _redis.get(f"chain_trace:{request_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return json.loads(raw)


@router.get("/traces")
def list_traces(limit: int = 20):
    """Retrieves metadata of the most recent traces recorded in Redis."""
    if _redis is None:
        return []
    try:
        # Retrieve recent request IDs
        ids = _redis.lrange("chain_traces_list", 0, limit - 1)
        traces = []
        for request_id in ids:
            raw = _redis.get(f"chain_trace:{request_id}")
            if raw:
                trace_data = json.loads(raw)
                # Keep payload light: return metadata summaries
                traces.append({
                    "request_id": trace_data.get("request_id"),
                    "goal": trace_data.get("goal"),
                    "start_time": trace_data.get("start_time"),
                    "total_latency_ms": trace_data.get("total_latency_ms"),
                    "satisfied": trace_data.get("verify", {}).get("satisfied", False),
                    "repair_rounds": trace_data.get("repair_rounds"),
                    "step_count": len(trace_data.get("results", [])),
                })
        return traces
    except Exception as e:
        logger.warning(f"failed to list traces: {e}")
        return []
