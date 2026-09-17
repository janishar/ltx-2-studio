#!/usr/bin/env bash
# Launch ltx studio on its own, from anywhere: under `helm dev`, which keeps
# everything ltx studio keeps in ./.helm through helmstudio's runtime SDK.
#
#   [HELMSTUDIO_REPO=<checkout>] [LTX_MODEL=<LTX-2.5 files>] [HELMSTUDIO_SDKS=<dir>] [LTX_DEBUGPY=<port>] bash web/run.sh [helm dev flags]
#   bash web/run.sh stop
#
# What each variable does: web/README.md, "Running".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SDKS="${HELMSTUDIO_SDKS:-/tmp/helmstudio-sdks}"
HELM="helm"
PIDFILE="$SDKS/ltx-studio.pid"

# Ends the helm dev this script last started, which stops the studio before it
# exits. A run starts by doing the same, so starting again is a restart.
stop() {
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q " dev -f helmstudio.yaml"; then
    echo "helm dev: stopping the ltx studio started before ($pid)"
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do sleep 0.2; done
  fi
  rm -f "$PIDFILE"
}
if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

if [ -n "${HELMSTUDIO_REPO:-}" ]; then
  HELM="$SDKS/bin/helm"
  echo "helm: building from $HELMSTUDIO_REPO"
  (cd "$HELMSTUDIO_REPO" && go build -o "$HELM" ./cmd/helm)
  # Not in uv.lock until it is on PyPI: `uv sync` without --inexact removes it again.
  # Built from a copy: setuptools leaves build/ and an .egg-info beside the source
  # it builds, and the checkout is not ltx studio's to write in.
  echo "helm-runtime-sdk: installing into .venv from $HELMSTUDIO_REPO"
  sdk="$SDKS/src/helm-runtime-sdk"
  rm -rf "$sdk" && mkdir -p "$sdk"
  cp -R "$HELMSTUDIO_REPO/packages/helm-runtime-sdk/python/"{pyproject.toml,README.md,LICENSE,helm_runtime_sdk} "$sdk"
  uv pip install --quiet --reinstall-package helm-runtime-sdk --python .venv/bin/python "$sdk"
fi

if [ -n "${LTX_MODEL:-}" ]; then
  echo "weights: linking $LTX_MODEL as $SDKS/weights/ltx-2.5"
  .venv/bin/python - "$LTX_MODEL" "$SDKS/weights/ltx-2.5" <<'PY'
import sys
from pathlib import Path

import yaml

model, weights = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
manifest = yaml.safe_load(Path("helmstudio.yaml").read_text())
for declared in next(weight for weight in manifest["weights"] if weight["name"] == "ltx")["files"]:
    found = sorted(model.rglob(Path(declared).name))
    if not found:
        sys.exit(f"weights: {Path(declared).name} is not under {model}")
    link = weights / declared
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(found[0])
PY
  set -- -link "ltx=$SDKS/weights/ltx-2.5" "$@"
fi

if ! command -v "$HELM" >/dev/null; then
  echo "helm is not installed: install it from helmstudio's releases, or set HELMSTUDIO_REPO to build it" >&2
  exit 1
fi
if ! .venv/bin/python -c "import helm_runtime_sdk" 2>/dev/null; then
  echo "helm-runtime-sdk is not installed in .venv: install it, or set HELMSTUDIO_REPO to install it from a checkout" >&2
  exit 1
fi

VENV=.venv
if [ -n "${LTX_DEBUGPY:-}" ]; then
  # helm dev hands the studio only a restricted environment, so a request to
  # listen for a debugger cannot reach it as a variable. helm dev is given an
  # environment of links to .venv's own commands instead, whose python runs
  # .venv's interpreter under debugpy; renders start that interpreter directly.
  if ! .venv/bin/python -c "import debugpy" 2>/dev/null; then
    echo "debugpy: installing into .venv"
    uv pip install --quiet --python .venv/bin/python debugpy
  fi
  VENV="$SDKS/ltx-studio-debugpy"
  rm -rf "$VENV" && mkdir -p "$VENV/bin"
  cp .venv/pyvenv.cfg "$VENV/"
  for entry in "$PWD"/.venv/bin/*; do ln -s "$entry" "$VENV/bin/"; done
  rm "$VENV/bin/python"
  printf '#!/bin/sh\nexec "%s" -Xfrozen_modules=off -m debugpy --listen "127.0.0.1:%s" "$@"\n' "$PWD/.venv/bin/python" "$LTX_DEBUGPY" >"$VENV/bin/python"
  chmod +x "$VENV/bin/python"
  echo "debugpy: the studio listens for a debugger on 127.0.0.1:$LTX_DEBUGPY"
fi

stop
mkdir -p "$SDKS"
echo $$ >"$PIDFILE"
exec "$HELM" dev -f helmstudio.yaml -venv "$VENV" "$@"
