# Repository Analysis Report

**Updated:** 2026-07-18  
**Repository:** `agentic-capstone`  
**Scope:** Application, reasoning chain, tests, browser interfaces, containers, and CI/CD

## 1. Executive summary

Agentic Capstone is a Python 3.11 FastAPI service that demonstrates two agent patterns:

1. Direct execution of a caller-selected calculator, clock, or weather tool.
2. A Gemini-powered, step-by-step ReAct loop that selects tools until a goal is satisfied.

Redis provides conversation and trace persistence. Two static browser interfaces provide chat
and observability experiences. Docker packages the service, while GitHub Actions tests it,
publishes an image to GHCR, deploys it to EC2, checks health, and attempts rollback on failure.

The repository contains 29 tracked files. It is a compact service rather than a deeply nested
or multi-entry-point system.

## 2. Architecture

```text
Browser
 ├─ /chat ────────┬─ /agent ───────── direct tools
 │                 └─ /chain/run ───── Gemini ReAct loop
 └─ /dashboard ───── /chain/traces ─── Redis traces

GitHub Actions → GHCR → EC2 Docker Compose → FastAPI + Redis
```

### Runtime modules

| Path | Responsibility |
|---|---|
| `app/main.py` | FastAPI application, direct tools, health/version routes, UI routes |
| `app/static/chat.html` | Direct-agent and reasoning-chain chat with session history |
| `app/static/dashboard.html` | Trace metrics, execution timeline, prompts, and API telemetry |
| `reasoning_chain/chain.py` | Gemini client, ReAct orchestration, retries, references, circuit breaker |
| `reasoning_chain/tools.py` | Instrumented calculator, timezone, and weather tools |
| `reasoning_chain/safe_math.py` | Bounded AST-based arithmetic evaluator |
| `reasoning_chain/schemas.py` | Pydantic contracts for plans, results, traces, and sessions |
| `reasoning_chain/router.py` | Chain, session, and trace APIs with Redis persistence |

## 3. Request workflows

### Direct tool call

`POST /agent` validates the tool name and argument, runs a direct tool, measures latency, and
returns `AgentResponse`. It does not call Gemini.

### Reasoning-chain call

`POST /chain/run` optionally loads clean role-based session memory from Redis. Gemini receives a
rolling summary, up to 16 recent `user`/`model` messages, the current goal, and prior tool results
within a configurable token budget. Full traces and tool telemetry remain outside conversational
context. Users can select a validated ReAct temperature from the chat UI, while summary generation
retains its separate server-controlled temperature. The effective value is captured in trace
telemetry. The service executes at most eight actions. Prior results resolve references such as
`[1]`, and a tool that fails twice is disabled for the rest of the run. The complete `ChainTrace`
is persisted separately for 24 hours, and capped session display messages are appended when a
`session_id` is supplied.

### Observability

`GET /chain/traces` returns recent trace summaries. `GET /chain/trace/{request_id}` returns the
full trace, including tool inputs and outputs, model prompts and responses, token usage, API-call
metadata, latency, and the final outcome. The dashboard calculates aggregate metrics in-browser.

## 4. API surface

The service exposes 14 application operations:

- Core: `/`, `/health`, `/version`, `/chat`, `/dashboard`, and `/agent`.
- Chain: `/chain/plan`, `/chain/run`, `/chain/traces`, `/chain/trace/{request_id}`,
  `/chain/sessions`, and three session metadata/message operations.

FastAPI additionally provides its standard OpenAPI and documentation routes.

## 5. Data and state

- Trace documents use `chain_trace:{request_id}` keys with a 24-hour expiry.
- `chain_traces_list` retains up to 100 recent request IDs.
- Session metadata and message lists are stored separately and currently have no expiry.
- When Redis is unavailable at import time, the application continues without persistence.
- Redis uses append-only persistence backed by the `redis_data` named volume.
- The chat also caches session metadata and the latest 50 messages per session in browser
  `localStorage` as a fallback when Redis is unavailable.

## 6. Delivery architecture

- The Dockerfile uses a two-stage Python 3.11 build and runs as a non-root user.
- Local Compose builds the application and starts Redis with health-based dependency ordering.
- Production Compose pulls the GHCR image and keeps Redis internal to the Compose network.
- CI runs Ruff, pytest on Python 3.11 and 3.12, and a Docker build check.
- CD preserves the previous `latest` image, publishes SHA and `latest` tags, deploys over SSH,
  performs an on-host health check, and rolls back to `previous-stable` after failure.

## 7. Security and reliability controls

- Arithmetic uses an AST allowlist with limits on expression size, syntax-tree size, exponent
  magnitude, integer size, and finite floating-point results.
- Weather API calls use HTTPS and redact the API key from trace URLs.
- Browser interfaces escape user, model, tool, and trace content before inserting dynamic HTML.
- Pydantic validates plans, tool names, results, and traces at orchestration boundaries.
- The ReAct loop is bounded, tool retries are bounded, and circuit-breaker state lasts for a run.
- Demo failure injection is off by default and controlled by environment variables.

## 8. Test and lint status

At the time of this report:

- `pytest -q`: 28 tests passed.
- `ruff check app reasoning_chain tests`: all checks passed.

Coverage includes core routes, direct tools, malformed expressions, timezone behavior, tool
retries, circuit breaking, reference resolution, bounded ReAct termination, session persistence,
trace listing, and mounted chain routes. LLM calls are mocked for deterministic CI.

## 9. Remaining risks and recommended work

1. Add authentication and authorization before exposing chat, sessions, or traces publicly.
2. Add request-size limits and rate limiting for LLM and external API cost control.
3. Avoid recording sensitive user content or full external API payloads without a retention and
   redaction policy.
4. Move Redis connection management into FastAPI lifespan handling and add reconnection logic.
5. Add session ownership; expiry and display message-count limits are now configurable.
6. Consolidate the duplicated direct and reasoning-chain tool implementations.
7. Move blocking Gemini and HTTP work off synchronous request handlers or use async clients.
8. Add browser-level tests for the chat and dashboard escaping and interaction paths.
9. Replace the image-tag rollback convention with immutable deployment manifests and verify the
   rolled-back service with a second health check.

## 10. Suggested reading order

1. `README.md`
2. `app/main.py`
3. `reasoning_chain/schemas.py`
4. `reasoning_chain/chain.py`
5. `reasoning_chain/tools.py` and `reasoning_chain/safe_math.py`
6. `reasoning_chain/router.py`
7. `tests/`
8. `app/static/`
9. Docker Compose and `.github/workflows/`

The screenshots under `docs/screenshots/` document the intended chat, trace, dashboard, and API
experiences. They are presentation evidence, while source and tests remain the correctness source
of truth.
