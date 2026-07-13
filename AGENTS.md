# Repository Guidelines

## Project Structure & Module Organization

- `app/main.py` contains the FastAPI service, tool implementations, and API routes.
- `tests/test_app.py` holds the pytest suite for health, root, and `/agent` behavior.
- `Dockerfile` and `docker-compose.yml` define the container and local stack setup.
- `requirements.txt` lists runtime dependencies; `requirements-dev.txt` adds test and lint tools.
- `pyproject.toml` stores Ruff and pytest configuration.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment.
- `pip install -r requirements-dev.txt` installs runtime, pytest, httpx, and Ruff.
- `uvicorn app.main:app --reload` runs the API locally with live reload.
- `pytest -v` runs the test suite.
- `ruff check app tests` checks formatting and import/style issues.
- `docker compose up --build` starts the app and Redis for the full local stack.

## Coding Style & Naming Conventions

- Use 4-space indentation and keep code compatible with Python 3.11.
- Follow Ruff rules already enabled in `pyproject.toml`: `E`, `F`, and `I`.
- Keep lines under 100 characters when practical.
- Use `snake_case` for functions, variables, and test names; use descriptive names for tools and API fields.
- Prefer small, explicit helper functions in `app/main.py` over hidden logic.

## Testing Guidelines

- Add tests under `tests/` with names that match `test_*.py`.
- Use `pytest` and `fastapi.testclient.TestClient` for API-level checks.
- Cover both success and failure paths for new tools or endpoints.
- Keep tests deterministic; mock or avoid external services unless the repo already provides a mock.

## Commit & Pull Request Guidelines

- The git history is minimal, so there is no strict established commit convention yet.
- Use short, imperative commit messages such as `feat: add weather tool` or `test: cover bad input`.
- PRs should include a clear summary, testing notes, and screenshots only when UI or docs output changes.
- Link related issues when available and call out any container, CI, or API contract changes.

## Security & Configuration Tips

- Do not commit secrets or host-specific values.
- Use environment variables such as `PORT` and `REDIS_URL` for runtime configuration.
- Keep the `/health` endpoint stable because it is used by container and deployment checks.
-