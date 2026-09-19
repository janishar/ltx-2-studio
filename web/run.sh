#!/usr/bin/env bash
# Launch ltx studio on its own, from anywhere: under `helm dev`, which keeps
# everything ltx studio keeps in ./.helm through helmstudio's runtime SDK.
#
#   [LTX_MODEL=<LTX-2.5 files>] [HELM=<helm>] [LTX_DEBUGPY=<port>] bash web/run.sh [helm dev flags]
#   bash web/run.sh stop
#
# helm comes from helmstudio's installer, and helm-runtime-sdk from PyPI, locked
# in uv.lock and installed by `uv sync`. What the script makes itself (the weight
# links, the debugger's environment, its pid file) stays in .cache/ltx-studio,
# beside the .helm helm dev keeps. What each variable does: web/README.md, "Running".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# helm dev records where a weight is linked, so the links stay in one place.
RUN="$PWD/.cache/ltx-studio"
HELM="${HELM:-helm}"
PIDFILE="$RUN/ltx-studio.pid"

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

if [ -n "${LTX_MODEL:-}" ]; then
  echo "weights: linking $LTX_MODEL as $RUN/weights/ltx-2.5"
  .venv/bin/python - "$LTX_MODEL" "$RUN/weights/ltx-2.5" <<'PY'
import sys
from pathlib import Path

import yaml

model, weights = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
manifest = yaml.safe_load(Path("helmstudio.yaml").read_text())
# Every declared weight, not just the base set: the optional ones (dev transformer,
# AV video VAE) are what --dev-transformer and --video-decoder diffusion need, and a
# model without them must still link, so a missing optional file is skipped.
for weight in manifest["weights"]:
    optional = weight.get("optional", False)
    for declared in weight["files"]:
        found = sorted(model.rglob(Path(declared).name))
        if not found:
            if optional:
                continue
            sys.exit(f"weights: {Path(declared).name} is not under {model}")
        link = weights / declared
        link.parent.mkdir(parents=True, exist_ok=True)
        link.unlink(missing_ok=True)
        link.symlink_to(found[0])
PY
  set -- -link "ltx=$RUN/weights/ltx-2.5" "$@"
fi

if ! command -v "$HELM" >/dev/null; then
  echo "helm is not installed. Install it with helmstudio's installer, or set HELM to its path:" >&2
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/janishar/helmstudio/main/installer/install.sh)"' >&2
  exit 1
fi
if ! .venv/bin/python -c "import helm_runtime_sdk" 2>/dev/null; then
  echo "helm-runtime-sdk is not installed in .venv: run uv sync" >&2
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
  VENV="$RUN/ltx-studio-debugpy"
  rm -rf "$VENV" && mkdir -p "$VENV/bin"
  cp .venv/pyvenv.cfg "$VENV/"
  for entry in "$PWD"/.venv/bin/*; do ln -s "$entry" "$VENV/bin/"; done
  rm "$VENV/bin/python"
  printf '#!/bin/sh\nexec "%s" -Xfrozen_modules=off -m debugpy --listen "127.0.0.1:%s" "$@"\n' "$PWD/.venv/bin/python" "$LTX_DEBUGPY" >"$VENV/bin/python"
  chmod +x "$VENV/bin/python"
  echo "debugpy: the studio listens for a debugger on 127.0.0.1:$LTX_DEBUGPY"
fi

stop
mkdir -p "$RUN"
echo $$ >"$PIDFILE"
exec "$HELM" dev -f helmstudio.yaml -venv "$VENV" "$@"
