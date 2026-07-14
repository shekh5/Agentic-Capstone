# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only in this stage (keeps final image small)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Create a non-root user (never run containers as root in production)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy only installed packages from builder stage, not build tools
COPY --from=builder /root/.local /home/appuser/.local
COPY app ./app
COPY reasoning_chain ./reasoning_chain

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
