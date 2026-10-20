# One image for both roles: the worker agent (default command) and the orchestrator
# (python -m rdas.live ...). The orchestrator talks to workers over HTTP only; the
# Docker socket is never mounted.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY workflows ./workflows
COPY configs ./configs

RUN useradd --uid 10001 --create-home rdas && mkdir -p /app/results && chown 10001 /app/results
USER 10001

ENV RDAS_HOST=0.0.0.0 RDAS_PORT=8000
EXPOSE 8000
CMD ["python", "-m", "rdas.workers.agent"]
