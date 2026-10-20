# Build the rdas image by streaming a tar of the build context to `docker build`.
#
# Why: on some Windows + Docker Desktop setups, BuildKit's file sync copies files from
# this folder as EMPTY files (seen on our machines: every .py in the image was 0 bytes).
# Sending the context as a tar archive bypasses that sync. On Linux/macOS plain
# `docker compose build` works; scripts/docker_build.sh is the equivalent helper.
#
#   powershell -ExecutionPolicy Bypass -File scripts\docker_build.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$tar = Join-Path $env:TEMP "rdas-context.tar"
tar -cf $tar --exclude=__pycache__ --exclude=*.egg-info Dockerfile pyproject.toml README.md src workflows configs
cmd /c "docker build -t rdas:latest - < `"$tar`""
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
Remove-Item $tar
docker run --rm --entrypoint python rdas:latest -c "import rdas.workers.agent as a; assert hasattr(a, 'main'); print('rdas:latest OK')"
