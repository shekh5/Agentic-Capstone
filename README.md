# Agentic Capstone: Tool-Calling Service + Full CI/CD Pipeline

A minimal FastAPI service that simulates an "agent" calling tools
(calculator, get_time, get_weather), wrapped in a complete
Git → Docker → GitHub Actions CI/CD pipeline.

This project exists to practice the **infrastructure** side of agentic
AI systems engineering — the agent logic is intentionally simple.

## Project structure

```
agentic-capstone/
├── app/
│   └── main.py              # FastAPI app + 3 tools
├── tests/
│   └── test_app.py          # pytest unit tests
├── .github/workflows/
│   ├── ci.yml                # lint + test + docker build check
│   └── cd.yml                # build, push to GHCR, deploy, rollback
├── Dockerfile                 # multi-stage, non-root, healthcheck
├── docker-compose.yml          # app + redis
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml             # ruff config
```

## Run it locally (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
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

## Run tests

```bash
pytest -v
ruff check app tests
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
- Add the Anthropic API as an actual reasoning/tool-selection layer
  instead of the client picking the tool directly
- Add structured logging + tracing for each tool call (important for
  debugging agent behavior in production)
- Add a staging environment + manual approval gate before prod deploy
