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

from typing import Optional

from fastapi import APIRouter, HTTPException

from .chain import decompose_goal, run_chain
from .schemas import ChainTrace, Plan, SessionMessage

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
def run(goal: str, session_id: Optional[str] = None):
    """Full plan -> execute -> verify -> repair loop using ReAct steps. 
    Accepts session_id to load conversation history memory."""
    context_str = ""
    if session_id and _redis:
        try:
            raw_msgs = _redis.lrange(f"session:{session_id}:messages", 0, -1)
            for raw in raw_msgs:
                msg = json.loads(raw)
                context_str += f"[{msg['sender'].capitalize()}]: {msg['text']}\n"
        except Exception as e:
            logger.warning(f"failed to load session context: {e}")

    try:
        trace = run_chain(goal, conversation_context=context_str if context_str else None)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chain failed: {e}") from e

    if session_id:
        trace.session_id = session_id

    _log_trace(trace)

    if session_id and _redis:
        try:
            user_msg = {
                "sender": "user",
                "text": goal,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            agent_msg = {
                "sender": "agent",
                "text": trace.verify.final_summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace": trace.model_dump()
            }
            _redis.rpush(f"session:{session_id}:messages", json.dumps(user_msg))
            _redis.rpush(f"session:{session_id}:messages", json.dumps(agent_msg))
            _redis.expire(f"session:{session_id}:messages", 7 * 24 * 3600)
        except Exception as e:
            logger.warning(f"failed to append session messages: {e}")

    return trace


@router.get("/session/{session_id}")
def get_session_messages(session_id: str):
    """Retrieves all message turns logged for a given session ID."""
    if _redis is None:
        return []
    try:
        raw_msgs = _redis.lrange(f"session:{session_id}:messages", 0, -1)
        return [json.loads(raw) for raw in raw_msgs]
    except Exception as e:
        logger.warning(f"failed to fetch session messages: {e}")
        return []


@router.post("/session/{session_id}/message")
def append_session_message(session_id: str, message: SessionMessage):
    """Logs an arbitrary chat message turn into a session's history log."""
    if _redis is None:
        return {"status": "error", "message": "Redis storage is not configured"}
    try:
        _redis.rpush(f"session:{session_id}:messages", message.model_dump_json())
        _redis.expire(f"session:{session_id}:messages", 7 * 24 * 3600)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"failed to append session message: {e}")
        return {"status": "error", "message": str(e)}


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
                    "total_tokens": trace_data.get("total_tokens", 0),
                })
        return traces
    except Exception as e:
        logger.warning(f"failed to list traces: {e}")
        return []
