#!/usr/bin/env sh
# Build the rdas image from a streamed tar context (see docker_build.ps1 for why).
set -e
cd "$(dirname "$0")/.."
tar -cf - --exclude=__pycache__ --exclude='*.egg-info' Dockerfile pyproject.toml README.md src workflows configs \
  | docker build -t rdas:latest -
docker run --rm --entrypoint python rdas:latest -c "import rdas.workers.agent as a; assert hasattr(a, 'main'); print('rdas:latest OK')"
