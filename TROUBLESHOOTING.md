# Troubleshooting & Resolution Log: Agentic Capstone

This document compiles all the compilation, linting, containerization, and API integration errors encountered during development, along with their root causes and resolutions.

---

## 1. CI Linting Failures (Ruff)

### Error
The CI build failed in the lint stage with two distinct issues:
*   `I001`: Unsorted imports in `tests/test_app.py`. Standard library imports must precede third-party libraries.
*   `E501`: Line too long in `tests/test_chain.py` (exceeded the 100-character configuration limit).

### Root Cause
*   The import block placed standard mock patching tools after FastAPI modules.
*   A single-line `PlanStep` definition exceeded 100 characters.

### Resolution
*   Reordered imports in `tests/test_app.py` to place standard library components first.
*   Wrapped the `PlanStep` arguments in `tests/test_chain.py` across multiple lines.

---

## 2. Docker container crash-loop (ModuleNotFoundError)

### Error
The application container on the EC2 host repeatedly failed to boot with the status `Restarting (1)`. Logs showed:
`ModuleNotFoundError: No module named 'reasoning_chain'`

### Root Cause
The `Dockerfile` copied only the `app/` folder into the image workdir, neglecting the `reasoning_chain/` orchestration package imported by `app/main.py`.

### Resolution
Added `COPY reasoning_chain ./reasoning_chain` to the runtime stage of the `Dockerfile` to ensure all source code packages are shipped inside the container image.

---

## 3. CD Deployment Directory Missing

### Error
The SSH deployment script crashed immediately upon trying to change directories.

### Root Cause
The script executed `cd ~/agentic-capstone` before verifying that the target workspace folder existed on the EC2 host.

### Resolution
Added `mkdir -p ~/agentic-capstone` as the very first command inside the SSH execution blocks.

---

## 4. CD Health Check Failures (Network Routing Block)

### Error
The health check pipeline step failed with a status code of `000000` (representing connection refused or timeout).

### Root Cause
*   **Network block**: The check was executed from the GitHub Actions runner (a public Microsoft Azure IP), which was blocked by the EC2's AWS Security Group configuration.
*   **Syntax error**: The script concatenated multiple `000` fallback codes when curl exited non-zero.

### Resolution
*   Modified `.github/workflows/cd.yml` to run the health check via SSH directly on the EC2 host. This allowed curling the container locally over `http://localhost:8000/health`.
*   Added a loop that retries the curl command 5 times (waiting 10 seconds between attempts) to handle container boot delays gracefully.

---

## 5. CD Rollback Failures (Missing Context)

### Error
When the health check failed, the rollback step executed but crashed immediately on Compose commands.

### Root Cause
*   The shell session on the EC2 host was missing the `GITHUB_REPOSITORY` environment variable, preventing the compose YAML file from resolving the target image name.
*   The runner lacked the credentials needed to pull the previous image from the GitHub Container Registry (GHCR).

### Resolution
Updated the rollback step script in `.github/workflows/cd.yml` to log into `ghcr.io` and export the `GITHUB_REPOSITORY` environment variable before executing `docker compose up`.

---

## 6. Client Closed Connection Error (Garbage Collection)

### Error
Calls to the reasoning chain failed with:
`RuntimeError: Cannot send a request, as the client has been closed`

### Root Cause
In `reasoning_chain/chain.py`, the client helper function `_get_client()` instantiated `genai.Client()` on every call but failed to store the reference in the global `_client` variable. As a result, the client object went out of scope, was garbage collected, and closed its connection pool.

### Resolution
Updated the getter function to store the instance in the global variable:
```python
_client = genai.Client()
```

---

## 7. Gemini Model 404 & 429 Errors (Quota and Availability)

### Error
*   `models/gemini-2.5-flash` returned a `404 NOT_FOUND` stating the model is deprecated for new keys.
*   `models/gemini-2.0-flash` returned `429 RESOURCE_EXHAUSTED` stating the free tier requests quota limit was set to 0.
*   `models/gemini-1.5-flash` was not recognized or found.

### Root Cause
*   Google deprecated `gemini-2.5-flash` preview versions.
*   The API key used was on a free tier where `2.0-flash` model usage was blocked.

### Resolution
*   Added a temporary `/list-models` diagnostics route to the live API to run test calls against candidate models directly on the EC2 instance.
*   Discovered that **`gemini-3.1-flash-lite`** was the only modern stable model that successfully generated a response.
*   Changed the default model string to `gemini-3.1-flash-lite` and removed the diagnostics route from `app/main.py`.

---

## 8. Pydantic Dict Validation Error (Argument Types)

### Error
The reasoning chain failed with:
`Pydantic validation error: steps.0.tool_input Input should be a valid dictionary`

### Root Cause
Unlike Anthropic Claude, `gemini-3.1-flash-lite` generated string arguments (e.g. `"Delhi"`) instead of dictionaries (e.g. `{"city": "Delhi"}`) for tool inputs.

### Resolution
Updated system instructions in `reasoning_chain/chain.py` for both the planning and verification models to explicitly enforce that `tool_input` must be a dictionary matching the tool's parameter names (e.g., `{"city": "..."}`). We provided concrete examples for each available tool.
