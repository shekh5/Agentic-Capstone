# Agentic Capstone: Tool-Calling Service + Full CI/CD Pipeline

A FastAPI agent service with direct tool execution, a Gemini-powered ReAct loop,
Redis-backed conversations and traces, browser interfaces, and a complete
Git → Docker → GitHub Actions → EC2 delivery pipeline.

This project demonstrates both agent orchestration and the infrastructure needed
to observe, test, package, and deploy it.

## Project structure

```
agentic-capstone/
├── app/
│   ├── main.py              # FastAPI app + 3 tools + UI routes
│   └── static/
│       ├── chat.html         # Interactive chat interface
│       └── dashboard.html    # Observability dashboard
├── reasoning_chain/
│   ├── chain.py              # ReAct loop orchestrator (Gemini)
│   ├── router.py             # /chain API routes + Redis persistence
│   ├── schemas.py            # Pydantic data contracts
│   └── tools.py              # Instrumented tools with failure injection
├── tests/
│   ├── test_app.py           # API endpoint tests
│   └── test_chain.py         # Reasoning chain logic tests
├── docs/
│   └── screenshots/          # UI and API screenshots
├── .github/workflows/
│   ├── ci.yml                # lint + test + docker build check
│   └── cd.yml                # build, push to GHCR, deploy, rollback
├── Dockerfile                 # multi-stage, non-root, healthcheck
├── docker-compose.yml          # local dev: app + redis
├── docker-compose.prod.yml     # production: GHCR image + redis
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml             # ruff + pytest config
```

## Screenshots

### Chat Interface (`/chat`)
The interactive chat UI supports two orchestration modes: **Direct Agent** (manual
tool selection) and **Reasoning Chain** (goal-based multi-step reasoning via Gemini).

<p align="center">
  <img src="docs/screenshots/chat-ui.jpg" alt="Chat UI — Dark glassmorphism interface with sidebar, mode selector, and chat threads" width="800"/>
</p>

### Reasoning Chain in Action
A multi-step conversation showing the ReAct loop: the agent decomposes a goal into
tool calls, executes them, and returns a verified answer with a collapsible
execution trace timeline.

<p align="center">
  <img src="docs/screenshots/chat-conversation.jpg" alt="Chat showing weather query with execution trace — weather tool then calculator" width="800"/>
</p>

### Observability Dashboard (`/dashboard`)
Real-time metrics (success rate, latency, token usage), a searchable trace history
table, and a detailed trace inspector panel for debugging agent behavior.

<p align="center">
  <img src="docs/screenshots/dashboard-overview.jpg" alt="Dashboard with metric cards, trace history table, and inspector panel" width="800"/>
</p>

### Trace Inspector — Execution Path
Drill into any trace to see the full execution timeline: plan decomposition, each
tool step with input/output/latency/tokens, API call details, and verification
outcome.

<p align="center">
  <img src="docs/screenshots/dashboard-trace-inspector.jpg" alt="Trace inspector showing step-by-step execution path with weather and calculator tools" width="800"/>
</p>

### API Documentation (`/docs`)
Auto-generated Swagger UI showing all 14 endpoints across the core service and
reasoning chain modules.

<p align="center">
  <img src="docs/screenshots/swagger-api-docs.jpg" alt="FastAPI Swagger UI showing all endpoints grouped by default and chain" width="800"/>
</p>

### API Endpoints
Live JSON responses from the health check and root service info endpoints.

<p align="center">
  <img src="docs/screenshots/api-endpoints.jpg" alt="Health and root endpoint JSON responses" width="800"/>
</p>

## Run it locally (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export GEMINI_API_KEY=your-key       # required for /chain routes
export WEATHER_API_KEY=your-key      # optional; otherwise deterministic mock weather is used
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for the interactive Swagger UI.

Try it:
```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"tool": "calculator", "argument": "2 + 2 * 10"}'
```

## Run it with Docker

```bash
docker build -t agentic-capstone .
docker run -p 8000:8000 agentic-capstone
```

## Run the full stack (app + redis) with Compose

```bash
docker compose up --build
```

Redis uses append-only persistence on the `redis_data` named volume, so sessions
and traces survive ordinary container recreation. The chat UI also caches the latest
50 messages per session in browser storage as a fallback. `docker compose down -v`
intentionally deletes the Redis volume.

## Run tests

```bash
pytest -v
ruff check app reasoning_chain tests
```

## Setting up the pipeline on GitHub

1. Create a new repo on GitHub and push this code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: agentic capstone scaffold"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **CI** (`ci.yml`) runs automatically on every push/PR — no setup needed.
   It lints with `ruff`, runs `pytest` across Python 3.11 and 3.12, and
   verifies the Docker image builds.

3. **CD** (`cd.yml`) runs on every push to `main`. It:
   - Builds and pushes your image to GitHub Container Registry (GHCR)
   - Tags it with both the git SHA (traceable) and `latest`
   - Runs a placeholder deploy step
   - Health-checks the deployed service
   - Rolls back automatically if the health check fails

   To make the deploy step real, add these **repo secrets**
   (Settings → Secrets and variables → Actions):
   - `APP_HEALTH_URL` — e.g. `https://your-app.fly.dev/health`
   - `DEPLOY_HOOK_URL` (if using Render) or configure `flyctl`/SSH auth
     depending on your host.

## Suggested learning path through this repo

1. Get it running locally, understand `main.py`
2. Make a change on a feature branch, open a PR, watch CI run
3. Intentionally break a test — watch CI fail and block the merge
4. Merge to `main` — watch CD build+push an image to GHCR
5. Pick a free host (Fly.io or Render) and wire up the real deploy step
6. Break the `/health` endpoint on purpose, deploy, and watch the
   rollback step trigger
7. Add a 4th tool to the agent and repeat the whole cycle

## Where this goes next (once comfortable)

- Swap the mock tools for real ones (weather API, real DB-backed memory)
- Consolidate the direct and reasoning-chain tool registries
- Add authentication, rate limiting, and access controls for traces
- Add a staging environment + manual approval gate before prod deploy


# Reasoning chain

The active orchestrator uses a bounded ReAct loop: Gemini proposes one validated
tool action, the service executes it, and the updated history is returned to Gemini
until the goal is satisfied or the eight-step limit is reached.

## Wire it in

```python
# main.py (or wherever your FastAPI app is created)
from reasoning_chain.router import router as chain_router
app.include_router(chain_router, prefix="/chain")
```

The tool layer uses a bounded arithmetic parser, IANA timezone handling, optional
WeatherAPI.com integration, retries, and a circuit breaker. Failure injection is
off by default and can be enabled with `WEATHER_FAILURE_RATE` or
`CALCULATOR_BAD_INPUT_RATE`, using values between `0` and `1`.

## Endpoints

- `POST /chain/plan?goal=...` — decomposition only, no tools run. Use this
  first to sanity-check the model's reasoning.
- `POST /chain/run?goal=...&session_id=...` — run the ReAct loop and return its trace.
- `GET /chain/traces` — list recent trace summaries.
- `GET /chain/trace/{request_id}` — replay a past run from Redis.
- `/chain/session/...` routes — save session metadata and conversation history.

## Try it locally

```bash
export GEMINI_API_KEY=...
uvicorn app.main:app --reload
curl -X POST "http://localhost:8000/chain/plan?goal=what+time+is+it+and+is+it+raining+in+Tokyo"
curl -X POST "http://localhost:8000/chain/run?goal=what+time+is+it+and+is+it+raining+in+Tokyo"
```

To demonstrate recovery behavior locally, set `WEATHER_FAILURE_RATE=0.35`.
Production deployments should leave failure injection unset or explicitly set to `0`.

## Tests

```bash
pip install pytest --break-system-packages
pytest tests/test_chain.py -v
```

All LLM calls are mocked, so this runs in your existing CI (Python
3.11/3.12 matrix) without needing an API key or network access.

## What to look at once it's running

1. **`/chain/plan`** — read the JSON. Does the model's decomposition make
   sense for a goal you didn't anticipate? This is where you'll spend most
   of your debugging time in real agent work.
2. **Force a failure** — run with `WEATHER_FAILURE_RATE=1.0` and hit
   `/chain/run`. Confirm the circuit breaker kicks in
   and `final_summary` is honest about what's missing, instead of the
   model quietly making up a temperature.
3. **Model-call telemetry** — inspect prompts, token usage, tool inputs, outputs,
   retries, and API latency in `/dashboard`.

## Next step (Phase 2 memory)

Session context and trace history are stored in Redis. A next memory phase would
add explicit retention limits, summarization, and user-level isolation.
