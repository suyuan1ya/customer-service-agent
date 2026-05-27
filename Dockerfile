FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY main.py config.py .
COPY agents/ agents/
COPY tools/ tools/
COPY orchestrator/ orchestrator/
COPY utils/ utils/
COPY knowledge_base/ knowledge_base/
COPY static/ static/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "python main.py"]
