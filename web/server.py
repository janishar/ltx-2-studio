#!/usr/bin/env python3
"""ltx studio — a local web control surface for every ``ltx-2-mlx`` command.

No web framework (no FastAPI/uvicorn), so it runs in the repo's existing uv
environment with helmstudio's runtime SDK. Each render spawns
``python -m ltx_pipelines_mlx <subcommand>`` with an argv built from the browser
form; one job runs at a time (one GPU) and the rest wait in a queue. Progress
streams to the browser over SSE; each render's log goes to helmstudio, whose
helm-terminal the page streams it with.

ltx studio keeps nothing of its own. ``State`` keeps and reads everything
through helmstudio's runtime SDK, and writes files only where helmstudio says;
web/README.md, "Where things are kept", lists where each thing goes. That holds
on its own too: a Python studio runs standalone under ``helm dev``, whose
provider keeps the same things in ``./.helm/`` (helmstudio's
docs/design/07-platform-services.md §8), so :func:`connect` refuses to start
without one. The SDK's same-origin proxy is mounted at ``/helm/``, through
which the page reaches helm-css, the browser runtime and components, the
studio's hue, the theme and the assets without ever holding the token.

Usage:
    helm dev -f helmstudio.yaml -venv .venv -link ltx=/path/to/LTX-2.5    (see web/run.sh)
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import ipaddress
import json
import mimetypes
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, ClassVar
from urllib.parse import parse_qs, unquote, urlparse

try:
    from helm_runtime_sdk import HelmError, from_env
    from helm_runtime_sdk.proxy import PREFIX, Proxy
except ImportError:  # connect() says what is missing; the tests stand in for the SDK
    HelmError = Exception
    from_env = None
    PREFIX = "/helm/"
    Proxy = None

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"

#: Subcommands the UI may run, and which of them take --model / --gemma / --quantize-on-load.
ALLOWED_COMMANDS = {
    "generate", "a2v", "retake", "extend", "keyframe", "ic-lora", "hdr-ic-lora", "lipdub",
    "enhance", "info", "preprocess", "slice", "train",
}  # fmt: skip
MODEL_COMMANDS = ALLOWED_COMMANDS - {"enhance", "slice", "train"}
GEMMA_COMMANDS = MODEL_COMMANDS - {"info"} | {"enhance"}
QUANTIZE_COMMANDS = {"generate", "a2v", "retake", "extend", "keyframe", "ic-lora", "hdr-ic-lora", "lipdub"}
SERVER_OWNED_FLAGS = {"--output", "-o", "--model", "-m", "--gemma", "--quantize-on-load", "--stepwise-image-output-dir"}
#: Subcommands whose pipelines accept the --stepwise-* live preview flags.
STEPWISE_COMMANDS = {"generate", "a2v", "retake", "extend", "keyframe", "ic-lora", "hdr-ic-lora", "lipdub"}
PREVIEW_NAME = re.compile(r"^seed_-?\d+(?:_s(\d+))?_step(\d+)of(\d+)\.webp$")

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _load(name: str, path: Path):
    """Import a module from its file, whether this server runs as a script or is loaded by the tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


#: scripts/ltx_run.py, for its model inspection (filesystem only).
LTX_RUN = _load("ltx_run", REPO_ROOT / "scripts" / "ltx_run.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_name(text: str, default: str = "untitled") -> str:
    cleaned = SAFE_NAME.sub("-", text.strip()).strip(".-")
    return cleaned[:80] or default


def which(tool: str) -> str | None:
    return shutil.which(tool)


def ffprobe(path: Path) -> dict[str, Any]:
    """Duration, dimensions, fps and audio presence of a media file (empty dict on failure)."""
    if not which("ffprobe"):
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout  # fmt: skip
    except (subprocess.SubprocessError, OSError):
        return {}
    data = json.loads(out or "{}")
    info: dict[str, Any] = {}
    with contextlib.suppress(KeyError, TypeError, ValueError):
        info["duration"] = round(float(data["format"]["duration"]), 3)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and "width" not in info:
            info["width"], info["height"] = stream.get("width"), stream.get("height")
            with contextlib.suppress(KeyError, ValueError, ZeroDivisionError, AttributeError):
                num, den = stream["r_frame_rate"].split("/")
                info["fps"] = round(float(num) / float(den), 3)
            with contextlib.suppress(KeyError, TypeError, ValueError):  # images have no nb_frames
                info["frames"] = int(stream["nb_frames"])
        if stream.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def media_kind(name: str) -> str:
    mime = mimetypes.guess_type(name)[0] or ""
    if name.lower().endswith((".safetensors", ".yaml", ".yml", ".txt", ".json")):
        return "file"
    return mime.split("/")[0] if mime.split("/")[0] in {"image", "video", "audio"} else "file"


def host_only(hostport: str) -> str:
    """``host`` from a ``Host`` header value (``host:port``, ``[::1]:port`` or bare)."""
    if hostport.startswith("["):
        return hostport[1 : hostport.find("]")] if "]" in hostport else hostport
    return hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport


def host_allowed(hostport: str, allowed: set[str]) -> bool:
    """IP addresses and ``localhost`` always pass; any other name must be allowlisted (blocks DNS rebinding)."""
    host = host_only(hostport).lower()
    if not host:
        return False
    if host == "localhost" or host in allowed:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def request_guard(method: str, path: str, headers: Any, allowed_hosts: set[str]) -> tuple[int, str] | None:
    """Refuse DNS-rebinding, cross-site and non-JSON state-changing requests.

    There is no authentication, so this is what stops a web page you visit from
    driving the studio: a forged POST either carries a foreign ``Origin`` or
    cannot set a JSON content type (or ``X-Filename``) without a CORS preflight,
    which this server never answers. Returns ``(status, message)`` to refuse, or
    ``None`` to allow.
    """
    host = headers.get("Host", "")
    if not host_allowed(host, allowed_hosts):
        return 403, f"unrecognized Host header; start ltx studio with --allow-host {host_only(host)} to allow it"
    if method in {"GET", "HEAD"}:
        return None
    origin = headers.get("Origin")
    if origin:
        parsed = urlparse(origin)
        if origin == "null" or parsed.netloc.lower() != host.lower():
            return 403, "cross-origin request refused"
    elif headers.get("Sec-Fetch-Site") == "cross-site":
        return 403, "cross-site request refused"
    if urlparse(path).path == "/api/upload":
        return None if headers.get("X-Filename") else (400, "X-Filename header is required")
    if urlparse(path).path.startswith(PREFIX):
        # helmstudio's proxy forwards the page's own requests, merge patches
        # included; Host and Origin were checked above.
        return None
    content_type = (headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        return 415, "Content-Type must be application/json"
    return None


# ---------------------------------------------------------------------------
# estimates, failure hints, setup checks
# ---------------------------------------------------------------------------

#: Task values that change how much work a render does (compared exactly for estimates).
WORKLOAD_VALUE = re.compile(r"steps|pipeline|teacache|topology|skipStage2|keyframes|^mode$|^cfg$|^stg$", re.IGNORECASE)


def workload(params: dict[str, Any]) -> dict[str, Any] | None:
    """Comparable workload of a take's saved form ``params`` (``snapshot()`` in app.js), or None."""
    task = params.get("taskId")
    common = params.get("common")
    if not task or not isinstance(common, dict):
        return None
    values = (params.get("values") or {}).get(task) or {}
    try:
        width, height = int(common.get("width") or 0), int(common.get("height") or 0)
        frames = int(common.get("frames") or 0)
    except (TypeError, ValueError):
        return None
    auto = bool(common.get("autoDuration"))
    preview = common.get("preview") if isinstance(common.get("preview"), dict) else {}
    key = {
        "task": task, "width": width, "height": height, "frames": "auto" if auto else frames,
        "lowRam": bool(common.get("lowRam")), "quantize": common.get("quantize"),
        "tiles": [common.get("tileFrames"), common.get("tileSpatial")], "preview": bool(preview.get("enabled")),
        "values": {k: v for k, v in sorted(values.items()) if WORKLOAD_VALUE.search(k) and not isinstance(v, (dict, list))},
    }  # fmt: skip
    return {"task": task, "key": json.dumps(key, sort_keys=True), "work": 0 if auto else width * height * frames}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def estimate_from(takes: Iterable[tuple[Any, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """Predict a render's wall time from finished takes, each ``(elapsed seconds, form params)``.

    The median of takes with the same workload when there are any (exact);
    otherwise the median of same-task takes scaled by pixels x frames. Empty
    when nothing fits.
    """
    target = workload(params)
    if target is None:
        return {}
    same, scaled = [], []
    for elapsed, take_params in takes:
        prior = workload(take_params or {})
        if not isinstance(elapsed, (int, float)) or elapsed <= 0 or prior is None:
            continue
        if prior["key"] == target["key"]:
            same.append(float(elapsed))
        elif prior["task"] == target["task"] and prior["work"] > 0:
            scaled.append(float(elapsed) * target["work"] / prior["work"])
    if same:
        return {"seconds": round(_median(same)), "samples": len(same), "exact": True}
    if scaled:
        return {"seconds": round(_median(scaled)), "samples": len(scaled), "exact": False}
    return {}


DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(h|min|s)\b")


def parse_duration(text: str) -> float | None:
    """Seconds from ``utils/estimate.py::format_duration`` output (``5 s``, ``1 min 30 s``, ``1 h 30 min``)."""
    parts = DURATION_PART.findall(text)
    if not parts:
        return None
    scale = {"h": 3600, "min": 60, "s": 1}
    return sum(float(value) * scale[unit] for value, unit in parts)


#: (pattern, advice) for known failure output. Advice only: the studio never changes settings for you.
FAILURE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Impacting ?Interactivity|MTLCommandBufferError|Command buffer execution failed: (?!Insufficient)"),
        "The macOS GPU watchdog stopped a long Metal command. Let the display sleep during the render, or lower "
        "LTX2_DIT_EVAL_EVERY / LTX2_GEMMA_EVAL_EVERY (they split GPU work into smaller command buffers).",
    ),
    (
        re.compile(r"(?i)insufficient memory|out of memory|failed to allocate|cannot allocate|\bOOM\b"),
        "Ran out of memory. Try Low RAM block streaming, int8/int4 quantize on load, tiling for large canvases, "
        "or a smaller canvas / fewer frames. Live preview also keeps the VAE decoder loaded.",
    ),
    (
        re.compile(r"no DurationHead|Pass num_frames explicitly"),
        "This model has no DurationHead (LTX-2.3 pack), so auto duration can't run — turn off Auto duration and set frames.",
    ),
    (
        re.compile(r"(?i)GatedRepoError|401 Client Error|Repository Not Found|RepositoryNotFoundError|gated repo"),
        "Hugging Face refused the download. Check the repo id, accept the model's terms on huggingface.co, "
        "and run `hf auth login` in the shell that starts the studio.",
    ),
    (
        re.compile(r"(?i)(dev.transformer|transformer-dev)\S*.*not found|not found.*(dev.transformer|transformer-dev)"),
        "This task needs the dev transformer, which the model directory doesn't have. Pick a distilled pipeline "
        "or point the model at a pack with transformer-dev.safetensors.",
    ),
    (
        re.compile(r"(?i)(ffmpeg|ffprobe) not found|ffmpeg.*(failed|error)"),
        "FFmpeg failed or wasn't found. Install it with `brew install ffmpeg` and restart the studio.",
    ),
    (
        re.compile(r"FileNotFoundError"),
        "A file the pipeline needed is missing. Open the Model dialog to check the model directory, and check "
        "that the task's inputs and LoRA paths still exist.",
    ),
]


def hint_for(tail: list[str]) -> str:
    """Advice for the first known failure pattern found in the output tail (newest lines first)."""
    for pattern, hint in FAILURE_HINTS:
        if any(pattern.search(line) for line in reversed(tail)):
            return hint
    return ""


def last_error_line(tail: list[str], returncode: int) -> str:
    """The line to show for a failed job: the last exception-looking line, else the last line."""
    for line in reversed(tail):
        text = line.strip()
        if re.match(r"^[A-Za-z_.]*(Error|Exception)\b.*:", text) or text.startswith("error:"):
            return text
    return next((line for line in reversed(tail) if line.strip()), f"exit code {returncode}")


def setup_checks(info: Any) -> list[dict[str, str]]:
    """Model and tool checks for the Model dialog. level: ok | warn | bad | info."""
    checks: list[dict[str, str]] = []
    if info is None:
        checks.append({"level": "bad", "label": "model", "detail": "not configured"})
    elif not info.local:
        path_like = info.model.startswith(("/", "~", "."))
        checks.append({
            "level": "bad" if path_like else "info", "label": "model",
            "detail": "directory not found" if path_like else "Hugging Face repo id — not inspected, downloaded on first use",
        })  # fmt: skip
    else:
        root = Path(info.model).expanduser()
        names = {p.name for p in root.rglob("*.safetensors")} if root.is_dir() else set()
        checks.append({"level": "ok", "label": "model", "detail": "LTX-2.5" if info.is_25 else "LTX-2.3 / other"})
        transformer = info.has_distilled or info.has_dev
        checks.append({
            "level": "ok" if transformer else "bad", "label": "transformer",
            "detail": " + ".join(n for n, ok in (("distilled", info.has_distilled), ("dev", info.has_dev)) if ok) or "none found",
        })  # fmt: skip
        official = any(n.startswith("ltx-2.5") for n in names)
        if not official:
            vae = any(n.startswith("vae_decoder") for n in names)
            checks.append(
                {
                    "level": "ok" if vae else "warn",
                    "label": "VAE decoder",
                    "detail": "found" if vae else "vae_decoder*.safetensors not found",
                }
            )
            upscaler = any(n.startswith("spatial_upscaler") for n in names)
            checks.append(
                {
                    "level": "ok" if upscaler else "warn",
                    "label": "upscaler",
                    "detail": "found" if upscaler else "no spatial_upscaler — two-stage pipelines will fail",
                }
            )
    for tool in ("ffmpeg", "ffprobe"):
        found = which(tool)
        checks.append(
            {"level": "ok" if found else "bad", "label": tool, "detail": found or "not on PATH — brew install ffmpeg"}
        )
    return checks


def checks_status(checks: list[dict[str, str]]) -> str:
    levels = {c["level"] for c in checks}
    return "bad" if "bad" in levels else "warn" if levels & {"warn", "info"} else "ok"


# ---------------------------------------------------------------------------
# helmstudio: its runtime SDK, through which ltx studio keeps everything
# ---------------------------------------------------------------------------

#: Where the page reads an asset: the studio API, through the proxy.
ASSETS = f"{PREFIX}api/v1/assets/"
#: A job's status in ltx studio, as a task job's state.
JOB_STATES = {"done": "succeeded", "failed": "failed", "cancelled": "cancelled"}
#: The kv document of the page's preferences: namespace and key.
PREFERENCES = ("ui", "preferences")
#: The records collection of the folders training tools write.
FOLDERS = "folders"


class UnavailableError(RuntimeError):
    """ltx studio was started without helmstudio, or without its runtime SDK."""


def connect() -> tuple[Any, Any]:
    """helmstudio's client and same-origin proxy, from the environment it starts ltx studio with.

    :class:`UnavailableError` without them.
    """
    if from_env is None:
        raise UnavailableError(
            "helm-runtime-sdk is not installed in this environment; ltx studio keeps everything through it"
        )
    if not os.environ.get("HELM_API"):
        raise UnavailableError(
            "helmstudio did not start ltx studio. Start it from helmstudio, or on its own with "
            "`helm dev -f helmstudio.yaml` (web/run.sh), which keeps what it stores in ./.helm"
        )
    return from_env(), Proxy.from_env()


def asset_url(asset_id: str) -> str:
    return f"{ASSETS}{asset_id}"


def thumb_url(asset_id: str) -> str:
    return f"{ASSETS}{asset_id}/thumb?w=320"


def without_nulls(value: Any) -> Any:
    """``value`` as a state document stores it: a merge patch removes a member whose value is null."""
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None}
    return value


def merge_patch(old: Any, new: Any) -> Any:
    """The JSON merge patch (RFC 7396) that turns ``old`` into ``new``; ``{}`` when they are equal.

    ``new`` holds no nulls (:func:`without_nulls`): in a patch, null removes.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return new
    patch: dict[str, Any] = {key: None for key in old if key not in new}
    for key, value in new.items():
        if key not in old:
            patch[key] = value
        elif old[key] != value:
            patch[key] = merge_patch(old[key], value) if isinstance(old[key], dict) else value
    return patch


def media_hints(probe: dict[str, Any]) -> dict[str, Any]:
    """What adopting a file may tell helmstudio about it, from ffprobe."""
    hints = {"width": probe.get("width"), "height": probe.get("height"),
             "duration_s": probe.get("duration"), "fps": probe.get("fps")}  # fmt: skip
    return {key: value for key, value in hints.items() if value}


def pages(fetch: Callable[..., dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Every item of a paged listing."""
    cursor = None
    while True:
        page = fetch(limit=200, cursor=cursor)
        yield from page.get("items") or []
        cursor = page.get("next_cursor")
        if not cursor:
            return


class HelmJob:
    """A render, reported to helmstudio as a task job: its state, its progress and its log.

    The log is what the page's terminal streams (helm-terminal), so it is sent
    on a timer rather than when the next line comes: a line printed before a
    minute of denoising shows at once. Reporting never fails a render. A call
    helmstudio refuses is printed, and the render goes on.
    """

    #: Seconds between progress reports.
    INTERVAL_S = 1.0
    #: Seconds between log appends while the render runs.
    LOG_INTERVAL_S = 0.5

    def __init__(self, client: Any, session_id: str, local_id: str) -> None:
        self.client = client
        self.local_id = local_id
        self.id = client.jobs.create({"state": "queued", "subject_kind": "session", "subject_id": session_id})["id"]
        self.cancelling = False
        self._lock = threading.Lock()
        self._sending = threading.Lock()
        self._unsent: list[str] = []
        self._ended = threading.Event()
        self._progressed = time.monotonic()

    def start(self) -> None:
        self._call(self.client.jobs.update, self.id, {"state": "running"})
        threading.Thread(target=self._send_log, name=f"helmstudio-log-{self.id}", daemon=True).start()

    def progress(self, percent: int) -> None:
        if time.monotonic() - self._progressed >= self.INTERVAL_S:
            self._progressed = time.monotonic()
            self._call(self.client.jobs.update, self.id, {"progress_num": percent, "progress_den": 100})

    def log(self, line: str, rewrites: bool = False) -> None:
        """Add a line to the log. A ``rewrites`` line is progress, which the terminal draws over the one before it.

        It is sent ending in a carriage return. Whatever is written after it
        draws over it, as on a terminal, so a progress line not sent yet gives
        way to the next line: of a bar's updates between two appends only the
        last is sent, and none when the bar closes with its final line.
        """
        text = f"{line}\r" if rewrites else line
        with self._lock:
            if self._unsent and self._unsent[-1].endswith("\r"):
                self._unsent[-1] = text
            else:
                self._unsent.append(text)

    def _send_log(self) -> None:
        while not self._ended.wait(self.LOG_INTERVAL_S):
            self.flush()

    def flush(self) -> None:
        """Append the lines not sent yet. One append at a time, so the log keeps their order."""
        with self._sending:
            with self._lock:
                unsent, self._unsent = self._unsent, []
            for start in range(0, len(unsent), 1000):
                batch = [line[:16384] for line in unsent[start : start + 1000]]
                self._call(self.client.jobs.append_log, self.id, {"lines": batch})

    def finish(self, status: str, error: str | None = None) -> None:
        self._ended.set()
        self.flush()
        body: dict[str, Any] = {"state": JOB_STATES.get(status, "failed")}
        if body["state"] == "succeeded":
            body.update(progress_num=100, progress_den=100)
        elif body["state"] == "failed":
            body["last_error"] = {"code": "render_failed", "message": (error or "the render failed")[:4000]}
        self._call(self.client.jobs.update, self.id, body)

    @staticmethod
    def _call(method: Callable[..., Any], *args: Any) -> None:
        try:
            method(*args)
        except Exception as exc:
            print(f"[helmstudio] {exc}", flush=True)


# ---------------------------------------------------------------------------
# state: the model, and everything helmstudio keeps for ltx studio
# ---------------------------------------------------------------------------


class State:
    """The model renders use, and everything ltx studio keeps, which it keeps and reads through helmstudio.

    ``client`` is the runtime SDK's client, and ``proxy`` its same-origin proxy
    (:func:`connect`).
    """

    def __init__(self, model: str, gemma: str | None, client: Any, proxy: Any) -> None:
        self.lock = threading.Lock()
        self.model = model
        self.gemma = gemma
        self.client = client
        self.proxy = proxy
        paths = client.me.get()["paths"]
        #: Scratch helmstudio gives this launch of the studio, and clears when it stops it.
        self.stage = Path(paths["stage"])
        #: The studio's own persistent directory, for files that are not assets.
        self.data = Path(paths["data"])
        #: Guards the sessions read from helmstudio and the renders reported to it.
        self._helm_lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}  # by name
        self._renders: dict[str, HelmJob] = {}  # renders queued or running, by ltx studio's job id
        self.active = self.last_opened() or next(iter(self.sessions()), "session-1")
        #: The render running now: lines go to its log.
        self.running: dict[str, Any] | None = None

    def model_info(self) -> dict[str, Any]:
        info = LTX_RUN.inspect_model(self.model) if self.model else None
        if info is None:
            return {"model": "", "configured": False, "checks": setup_checks(None), "status": "bad"}
        checks = setup_checks(info)
        return {
            "model": self.model,
            "configured": True,
            "local": info.local,
            "exists": Path(self.model).expanduser().exists() or not info.local,
            "has_distilled": info.has_distilled,
            "has_dev": info.has_dev,
            "is_25": info.is_25,
            "gemma": self.gemma or "",
            "checks": checks,
            "status": checks_status(checks),
        }

    # sessions -------------------------------------------------------------
    def session_name(self, requested: Any) -> str:
        """The session a request names, or the active one."""
        return str(requested or self.active).strip() or "session-1"

    def sessions(self) -> list[str]:
        """Every session's name, read again from helmstudio."""
        with self._helm_lock:
            self._sessions = {session["name"]: session for session in pages(self.client.sessions.list)}
            return sorted(self._sessions)

    def last_opened(self) -> str | None:
        """The session opened most recently, when one has been opened."""
        latest = next(iter(self.client.sessions.list(limit=1).get("items") or []), None)
        return latest["name"] if latest and latest.get("opened_at") else None

    def session(self, name: str, *, create: bool = False) -> dict[str, Any] | None:
        """The live session called ``name``; made when there is none and ``create`` is set."""
        with self._helm_lock:
            if name not in self._sessions:
                self.sessions()
            if name not in self._sessions and create:
                self._sessions[name] = self.client.sessions.create({"name": name, "state": {}})
            return self._sessions.get(name)

    def _existing(self, name: str) -> dict[str, Any]:
        session = self.session(name)
        if session is None:
            raise ValueError(f"no session named {name!r}")
        return session

    def activate(self, name: str) -> dict[str, Any]:
        """Open a session, making it when there is none, so that it is the one opened most recently."""
        session = self.session(name, create=True)
        self.client.sessions.activate(session["id"])
        self.active = session["name"]
        return {"name": session["name"], "settings": self.settings(session["name"])}

    def duplicate_session(self, session: str, new_name: str) -> dict[str, Any]:
        """Copy a session's settings and inputs under a new name, and open the copy. Its takes stay with the original."""
        if not new_name.strip():
            raise ValueError("enter a name")
        source = self._existing(session)
        try:
            copy = self.client.sessions.duplicate(source["id"], {"name": new_name.strip()})
        except HelmError as exc:
            if exc.status == 409:
                raise ValueError("a session with that name already exists") from None
            raise
        with self._helm_lock:
            self._sessions[copy["name"]] = copy
        return self.activate(copy["name"])

    def delete_session(self, session: str) -> dict[str, Any]:
        """Delete a session with its settings and inputs, and open another. Its takes stay in helmstudio's gallery."""
        self.client.sessions.delete(self._existing(session)["id"])
        with self._helm_lock:
            self._sessions.pop(session, None)
        remaining = self.sessions()
        return self.activate(remaining[0] if remaining else "session-1")

    def _state(self, name: str) -> dict[str, Any]:
        """A session's state document; empty when there is no such session."""
        return (self.session(name) or {}).get("state") or {}

    def settings(self, name: str) -> dict[str, Any]:
        return self._state(name).get("settings") or {}

    def save_settings(self, name: str, settings: dict[str, Any]) -> None:
        wanted = without_nulls(settings)

        def change(state: dict[str, Any]) -> dict[str, Any]:
            patch = merge_patch(state.get("settings") or {}, wanted)
            return {"settings": patch} if patch else {}

        self._update_state(name, change)

    def inputs(self, name: str) -> dict[str, dict[str, Any]]:
        """A session's inputs by name: an asset each, with what ltx studio knows about it."""
        return dict(self._state(name).get("inputs") or {})

    def put_input(self, session: str, name: str, entry: dict[str, Any], *, replace: bool = False) -> str:
        """Add an input to a session as ``name``, replacing one of that name when ``replace`` is set and
        otherwise taking a free variant of the name. Returns the name it got."""
        chosen = name

        def change(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal chosen
            taken = state.get("inputs") or {}
            stem, suffix = os.path.splitext(name)
            chosen = name
            while chosen in taken and not replace:
                chosen = f"{stem}-{uuid.uuid4().hex[:4]}{suffix}"
            patch = merge_patch(taken.get(chosen) or {}, without_nulls(entry))
            return {"inputs": {chosen: patch}} if patch else {}

        self._update_state(session, change)
        return chosen

    def _update_state(self, name: str, change: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Merge ``change(state)`` into a session's state, reading it again when another writer got there first."""
        with self._helm_lock:
            for attempt in range(3):
                session = self.session(name, create=True)
                patch = change(session.get("state") or {})
                if not patch:
                    return
                try:
                    updated = self.client.sessions.update(session["id"], {"state": patch}, if_match=session["etag"])
                except HelmError as exc:
                    if exc.status not in (404, 409) or attempt == 2:
                        raise
                    self._sessions.pop(name, None)
                    continue
                self._sessions[updated["name"]] = updated
                return

    # the page's preferences -----------------------------------------------
    def preferences(self) -> dict[str, Any]:
        """The page's preferences: the side panel, the terminal's height, notifications."""
        try:
            return self.client.kv.get(*PREFERENCES)["doc"]
        except HelmError as exc:
            if exc.status == 404:
                return {}
            raise

    def save_preferences(self, changes: dict[str, Any]) -> None:
        """Merge ``changes`` into the page's preferences; a null removes one."""
        try:
            self.client.kv.patch(*PREFERENCES, changes)
        except HelmError as exc:
            if exc.status != 404:
                raise
            self.client.kv.put(*PREFERENCES, without_nulls(changes))

    # the terminal: the log of the session's latest render -------------------
    def render_log(self, session: str, line: str, rewrites: bool = False, kind: str | None = None) -> None:
        """Add a line to the log of the session's running render, which helmstudio keeps and the terminal streams."""
        running = self.running
        report = self.render_report(running["id"]) if running and running["session"] == session else None
        if report is not None:
            colour = LOG_COLOURS.get(kind or "")
            report.log(f"\x1b[{colour}m{line}\x1b[0m" if colour else line, rewrites)

    def terminal_job(self, session: str) -> str | None:
        """The helmstudio job whose log the session's terminal shows: its latest render no longer queued.

        Looked for among helmstudio's latest jobs.
        """
        found = self.session(session)
        if found is None:
            return None
        for job in self.client.jobs.list(limit=100).get("items") or []:
            ours = (job.get("kind"), job.get("subject_kind"), job.get("subject_id")) == ("task", "session", found["id"])
            if ours and job.get("state") != "queued":
                return job["id"]
        return None

    # assets ---------------------------------------------------------------
    def staged(self, kind: str, name: str) -> Path:
        """A new path for a file named ``name`` in the stage directory, in a directory of its own."""
        path = self.stage / kind / uuid.uuid4().hex / Path(name).name
        path.parent.mkdir(parents=True)
        return path

    def adopt_staged(self, path: Path, kind: str, probe: dict[str, Any], *, pinned: bool = False) -> dict[str, Any]:
        """Adopt a file from :meth:`staged`, and remove the directory it had."""
        asset = self.adopt(path, kind, probe, pinned=pinned)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
        return asset

    def adopt(self, path: Path, kind: str, probe: dict[str, Any], *, pinned: bool = False) -> dict[str, Any]:
        """Adopt a file into the asset store by hardlink, never a copy.

        From the stage directory, the stage entry is then gone; from the data
        directory, the file stays where it is, read-only.
        """
        return self.client.assets.adopt({"path": str(path), "kind": kind, "pinned": pinned, **media_hints(probe)})

    def materialise(self, asset_id: str, name: str) -> Path:
        """An asset as a file a pipeline or ffmpeg can read, fetched into the stage directory once per launch."""
        path = self.stage / "assets" / asset_id / Path(name).name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.part")
            partial.write_bytes(self.client.assets.read(asset_id).read())
            partial.replace(path)
        return path

    def pin(self, asset_id: str, name: str, kind: str) -> None:
        """Pin an asset already stored: adopting its bytes again stores nothing, and pins it."""
        link = self.staged("pin", name)
        os.link(self.materialise(asset_id, name), link)
        self.adopt_staged(link, kind, {}, pinned=True)

    # renders --------------------------------------------------------------
    def input_path(self, session: str, name: str) -> Path:
        """The file a render reads for one of the session's inputs: its asset, fetched into the stage directory."""
        entry = self.inputs(session).get(name)
        if entry is None:
            raise ValueError(f"input not found in session: {name}")
        return self.materialise(entry["asset_id"], name)

    def outputs_dir(self, session: str, kind: str) -> Path:
        """Where a render writes: a take (``mp4``) to the stage directory, a folder (``dir``) to the data directory."""
        if kind == "dir":
            directory = self.data / "outputs" / self.session(session, create=True)["id"]
        else:
            directory = self.stage / "takes"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def previews_dir(self, session: str) -> Path:
        return self.stage / "previews"

    def take_finished(self, job: dict[str, Any], output: Path, previews: list[Path]) -> str:
        """Adopt the take and record it in the gallery, with its settings and the inputs it came from.

        Not its argv: absolute paths to the model and the checkout do not belong
        in a gallery row (helmstudio's docs/design/08-h3-dry-run.md).
        """
        session, probe = job["session"], ffprobe(output)
        entries = self.inputs(session)
        inputs = [{"asset_id": entries[name]["asset_id"], "role": role}
                  for name, role in job.get("inputs") or [] if name in entries]  # fmt: skip
        settings = job["params"] or {}
        params = {
            "name": output.name, "task_id": job["task_id"], "label": job["label"],
            "prompt": (settings.get("common") or {}).get("prompt"), "seed": job.get("seed"),
            "created": job["created"], "elapsed": job["elapsed"], "probe": probe,
            "settings": settings, "previews": [_rel(path, self.stage) for path in previews],
        }  # fmt: skip
        params = {key: value for key, value in params.items() if value not in (None, "", [])}
        asset = self.adopt(output, "video", probe)
        item = self.client.gallery.add({
            "kind": "video", "asset_id": asset["id"], "title": output.stem,
            "session_id": self.session(session, create=True)["id"], "params": params, "inputs": inputs,
        })  # fmt: skip
        return f"[helmstudio] kept {output.name} as gallery item {item['id']}"

    def folder_finished(self, job: dict[str, Any], folder: Path) -> str:
        """Keep the folder a training tool wrote: each file an asset, adopted where it is, and the folder a record.

        The folder stays in the data directory, where the next tool reads it by
        path; adopted, its files are read-only.
        """
        files = {}
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink():
                kind = media_kind(path.name)
                asset = self.adopt(path, kind if kind != "file" else "other", {})
                files[path.relative_to(folder).as_posix()] = asset["id"]
        record = self.client.records.insert(FOLDERS, {
            "session_id": self.session(job["session"], create=True)["id"],
            "name": folder.name, "path": folder.relative_to(self.data).as_posix(), "task_id": job["task_id"],
            "label": job["label"], "created": job["created"], "files": files,
        })  # fmt: skip
        return f"[helmstudio] kept {folder.name}: {len(files)} files as assets, in record {record['id']}"

    def job_queued(self, job: dict[str, Any]) -> None:
        """Report a render as a task job of its session. ``helm_job`` is its id, or None when helmstudio refused it."""
        job["helm_job"] = None
        try:
            report = HelmJob(self.client, self.session(job["session"], create=True)["id"], job["id"])
        except Exception as exc:  # reporting a render is never a condition of running it
            print(f"[helmstudio] could not report render {job['id']}: {exc}", flush=True)
            return
        with self._helm_lock:
            self._renders[job["id"]] = report
        job["helm_job"] = report.id

    def render_report(self, local_id: str) -> HelmJob | None:
        """The job a queued or running render is reported as; None when helmstudio could not be told of it."""
        return self._renders.get(local_id)

    def job_started(self, job: dict[str, Any]) -> None:
        self.running = job
        report = self.render_report(job["id"])
        if report is not None:
            report.start()

    def job_progress(self, job: dict[str, Any], progress: dict[str, Any]) -> None:
        report = self.render_report(job["id"])
        if report is not None:
            report.progress(percent(progress))

    def job_finished(self, job: dict[str, Any]) -> None:
        if self.running is job:
            self.running = None
        report = self.render_report(job["id"])
        if report is None:
            return
        report.finish(job["status"], job.get("error"))
        with self._helm_lock:
            self._renders.pop(job["id"], None)

    def follow_cancellations(self, cancel: Callable[[str], Any]) -> None:
        """Cancel a render, by ltx studio's job id, when helmstudio asks for it to be cancelled."""

        def follow() -> None:
            last = None
            while True:
                try:
                    for event in self.client.events.subscribe(last_event_id=last):
                        last = event.id or last
                        if event.name != "job":
                            continue
                        reported = (event.json() or {}).get("job") or {}
                        job = next((r for r in list(self._renders.values()) if r.id == reported.get("id")), None)
                        if job is not None and reported.get("cancel_requested_at") and not job.cancelling:
                            job.cancelling = True
                            cancel(job.local_id)
                except Exception:  # the stream ended: helmstudio restarted, or is stopping the studio
                    pass
                time.sleep(2)

        threading.Thread(target=follow, name="helmstudio-events", daemon=True).start()

    # takes ----------------------------------------------------------------
    @staticmethod
    def _take_name(item: dict[str, Any]) -> str:
        return (item.get("params") or {}).get("name") or f"{item.get('title') or item['id']}.mp4"

    def _take(self, item: dict[str, Any]) -> dict[str, Any]:
        """A gallery item, as the page shows a take."""
        params, asset = item.get("params") or {}, item.get("asset") or {}
        return {
            "name": self._take_name(item), "kind": "video", "size": asset.get("bytes"),
            "url": asset_url(item["asset_id"]), "thumb": thumb_url(item["asset_id"]),
            "task_id": params.get("task_id"), "label": params.get("label"),
            "created": params.get("created") or item["created_at"], "elapsed": params.get("elapsed"),
            "seed": params.get("seed"), "params": params.get("settings"), "probe": params.get("probe") or {},
            "starred": bool(item.get("starred")),
            # Live previews are scratch: a take from before helmstudio last started the studio has none left.
            "previews": [p for p in params.get("previews") or [] if (self.stage / p).is_file()],
        }  # fmt: skip

    def _takes(self, session: str) -> list[dict[str, Any]]:
        """A session's takes, newest first: its video items in helmstudio's gallery."""
        found = self.session(session)
        if found is None:
            return []
        return list(pages(lambda **page: self.client.gallery.query(session_id=found["id"], kind="video", **page)))

    def _find_take(self, session: str, name: str) -> dict[str, Any]:
        item = next((item for item in self._takes(session) if self._take_name(item) == name), None)
        if item is None:
            raise ValueError("take not found")
        return item

    def list_takes(self, session: str) -> list[dict[str, Any]]:
        return [self._take(item) for item in self._takes(session)]

    def set_star(self, session: str, name: str, starred: bool) -> dict[str, Any]:
        self.client.gallery.update(self._find_take(session, name)["id"], {"starred": bool(starred)})
        return {"ok": True, "starred": bool(starred)}

    def delete_take(self, session: str, name: str) -> None:
        """Delete a take from the gallery. helmstudio reclaims its file once nothing uses it."""
        self.client.gallery.delete(self._find_take(session, name)["id"])

    def estimate(self, session: str, params: dict[str, Any]) -> dict[str, Any]:
        """Predict a render's wall time from this session's takes (``estimate_from``)."""
        takes = [item.get("params") or {} for item in self._takes(session)]
        return estimate_from(((take.get("elapsed"), take.get("settings")) for take in takes), params)

    # inputs ---------------------------------------------------------------
    def list_inputs(self, session: str) -> list[dict[str, Any]]:
        entries = sorted(self.inputs(session).items(), key=lambda pair: pair[1].get("added") or "", reverse=True)
        items = []
        for name, entry in entries:
            item = {"name": name, "kind": entry.get("kind", "file"), "url": asset_url(entry["asset_id"]),
                    "original_name": entry.get("original_name", name), "probe": entry.get("probe") or {}}  # fmt: skip
            if item["kind"] == "video":
                item["thumb"] = thumb_url(entry["asset_id"])
            items.append(item)
        return items

    def _add_input(self, session: str, path: Path, original_name: str, *, replace: bool = False) -> dict[str, Any]:
        """Adopt a staged file as a pinned asset, and add it to the session's inputs.

        An upload of a name already taken gets a free variant of it; a frame or
        the audio of a take replaces the one extracted before (``replace``).
        """
        kind = media_kind(path.name)
        probe = ffprobe(path) if kind != "file" else {}
        asset = self.adopt_staged(path, kind if kind != "file" else "other", probe, pinned=True)
        entry = {"asset_id": asset["id"], "kind": kind, "original_name": original_name,
                 "probe": probe, "added": now_iso()}  # fmt: skip
        return {"name": self.put_input(session, path.name, entry, replace=replace), "kind": kind,
                "original_name": original_name, "probe": probe}  # fmt: skip

    def save_upload(self, session: str, original: str, body: BinaryIO, length: int) -> dict[str, Any]:
        stem, suffix = os.path.splitext(original)
        staged = self.staged("uploads", f"{safe_name(stem, 'upload')}{suffix.lower()}")
        receive(body, length, staged)
        return self._add_input(session, staged, original)

    def delete_input(self, session: str, name: str) -> None:
        """Take an input out of the session. Its asset stays pinned: takes made from it name it as an input."""

        def change(state: dict[str, Any]) -> dict[str, Any]:
            return {"inputs": {name: None}} if name in (state.get("inputs") or {}) else {}

        self._update_state(session, change)

    def frame(self, session: str, take: str, position: str) -> dict[str, Any]:
        """A take's first or last frame, as an input."""
        source = self.materialise(self._find_take(session, take)["asset_id"], take)
        dst = self.staged("frames", f"{Path(take).stem}-{position}.png")
        grab_frame(source, dst, position)
        return {"name": self._add_input(session, dst, dst.name, replace=True)["name"]}

    def audio(self, session: str, take: str) -> dict[str, Any]:
        """A take's audio track, as an input."""
        source = self.materialise(self._find_take(session, take)["asset_id"], take)
        dst = self.staged("audio", f"{Path(take).stem}-audio.wav")
        grab_audio(source, dst)
        return {"name": self._add_input(session, dst, dst.name, replace=True)["name"]}

    def use_video(self, session: str, kind: str, name: str) -> dict[str, Any]:
        """A take or an exported sequence, as an input: the same asset, pinned now that a session uses it."""
        if kind == "outputs":
            item = self._find_take(session, name)
            probe = (item.get("params") or {}).get("probe") or {}
        elif kind == "timeline":
            item = self._find_export(name)
            probe = self._export(item)["probe"]
        else:
            raise ValueError("invalid media kind")
        self.pin(item["asset_id"], name, "video")
        entry = {"asset_id": item["asset_id"], "kind": "video", "original_name": name,
                 "probe": probe, "added": now_iso()}  # fmt: skip
        return {"name": self.put_input(session, name, entry)}

    # timeline: sequences exported from helmstudio's timeline --------------
    def _export(self, item: dict[str, Any]) -> dict[str, Any]:
        asset = item.get("asset") or {}
        return {
            "name": f"{safe_name(item.get('title') or 'sequence', 'sequence')}-{item['id'][-6:].lower()}.mp4",
            "kind": "video", "size": asset.get("bytes"), "created": item["created_at"],
            "url": asset_url(item["asset_id"]), "thumb": thumb_url(item["asset_id"]),
            "probe": {"duration": asset.get("duration_s"), "width": asset.get("width"),
                      "height": asset.get("height"), "fps": asset.get("fps")},
        }  # fmt: skip

    def _exports(self) -> list[dict[str, Any]]:
        """The sequences exported from helmstudio's timeline, newest first."""
        items = pages(lambda **page: self.client.gallery.query(kind="video", **page))
        return [item for item in items if item.get("timeline_id")]

    def _find_export(self, name: str) -> dict[str, Any]:
        item = next((item for item in self._exports() if self._export(item)["name"] == name), None)
        if item is None:
            raise ValueError("sequence not found")
        return item

    def list_timeline(self, session: str) -> list[dict[str, Any]]:
        """Every exported sequence. A sequence belongs to no session, so each session lists them all."""
        return [self._export(item) for item in self._exports()]

    def delete_timeline(self, session: str, name: str) -> None:
        self.client.gallery.delete(self._find_export(name)["id"])

    # files ----------------------------------------------------------------
    def resolve_stage_path(self, rel: str) -> Path:
        """A live preview's path, relative to the stage directory, refusing anything that escapes it."""
        root = self.stage.resolve()
        path = (root / unquote(rel).lstrip("/")).resolve()
        if path != root and root not in path.parents:
            raise ValueError("path escapes the stage directory")
        return path


# ---------------------------------------------------------------------------
# runner: queue + subprocess + progress parsing
# ---------------------------------------------------------------------------

PHASE_RE = re.compile(r"^\[([^\]]+)\] \.\.\.$")
ESTIMATE_RE = re.compile(r"^\[estimate\] denoising(?: \(([^)]+)\))?: (\d+) steps")
REMAINING_RE = re.compile(r"^\[estimate\] [^:]+: ~(.+?) remaining")
TQDM_RE = re.compile(r"(Denoising[^:]*):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)")
SAVED_RE = re.compile(r"Saved to: (.+)$")
#: Stepper stages in order; a job's stage only ever moves forward (two-stage runs reload the transformer).
STAGES = ("encode", "load", "denoise", "decode", "save")
#: ANSI colours for ltx studio's own lines in a render's log, by kind; helm-terminal draws them in the log tokens.
LOG_COLOURS = {"cmd": "36", "done": "36", "failed": "31", "cancelled": "31", "err": "31"}
PHASE_STAGES = (
    (re.compile(r"^(Loading text encoder|Encoding prompt)"), "encode"),
    (re.compile(r"^Loading transformer"), "load"),
    (re.compile(r"^(Loading decoders|Decoding)"), "decode"),
)
#: A flag naming an input, as the role of that input in a take's provenance.
ROLE_FLAG = re.compile(r"--([a-z][a-z0-9-]{0,62})")


def percent(progress: dict[str, Any]) -> int:
    """A job's overall progress, 0-100, weighing its stages as the page's progress does (``overallPercent``)."""
    denoise = (75, 88) if (progress.get("stage") or 1) > 1 else (15, 75)
    bands = {"encode": (0, 8), "load": (8, 15), "denoise": denoise, "decode": (88, 98), "save": (98, 100)}
    low, high = bands.get(progress.get("stage_key") or "", (0, 0))
    if progress.get("stage_key") == "denoise" and progress.get("total"):
        return round(low + (high - low) * min(1, progress.get("step", 0) / progress["total"]))
    return low


def input_roles(args: list[Any]) -> list[tuple[str, str]]:
    """Each input a render's arguments name, with the flag before it as its role (``--image`` → ``image``)."""
    roles, role = [], "input"
    for token in args:
        if isinstance(token, str) and (flag := ROLE_FLAG.fullmatch(token)):
            role = flag.group(1).replace("-", "_")
        elif isinstance(token, dict) and "input" in token:
            roles.append((str(token["input"]), role))
    return roles


class Runner:
    def __init__(self, state: State) -> None:
        self.state = state
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pending: list[str] = []
        self.current: str | None = None
        self.proc: subprocess.Popen | None = None
        self.cond = threading.Condition()
        self.subscribers: list[queue.Queue] = []
        self.sub_lock = threading.Lock()
        #: Guards job dicts: the worker thread mutates them while request threads serialize summaries.
        self.job_lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        state.follow_cancellations(self.cancel)

    # events ---------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.sub_lock, contextlib.suppress(ValueError):
            self.subscribers.remove(q)

    def emit(self, kind: str, payload: Any) -> None:
        message = json.dumps({"type": kind, "data": payload})
        with self.sub_lock:
            for q in list(self.subscribers):
                with contextlib.suppress(queue.Full):
                    q.put_nowait(message)

    def log(self, session: str, line: str, replace: bool = False, kind: str | None = None) -> None:
        """Add a line to the log of the session's render, which the page's terminal streams from helmstudio."""
        self.state.render_log(session, line, replace, kind)

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "session", "task_id", "label", "status", "created", "started", "finished",
                "elapsed", "output", "returncode", "seed", "argv_display", "error", "hint",
                "preview_latest", "preview_count", "estimate_s", "helm_job")  # fmt: skip
        with self.job_lock:
            out = {k: job.get(k) for k in keys}
            out["progress"] = dict(job.get("progress") or {})
        return out

    def queue_state(self) -> list[dict[str, Any]]:
        with self.cond:
            ids = ([self.current] if self.current else []) + list(self.pending)
            return [self.summary(self.jobs[i]) for i in ids if i in self.jobs]

    # submission -----------------------------------------------------------
    def build_argv(self, req: dict[str, Any], job_id: str | None = None) -> tuple[list[str], Path | None, str]:
        """The argv for a render, the path it writes, and its session's name."""
        subcommand = req.get("subcommand")
        if subcommand not in ALLOWED_COMMANDS:
            raise ValueError(f"unsupported command: {subcommand!r}")
        session = self.state.session_name(req.get("session"))
        argv: list[str] = [subcommand]
        skip_next = False
        for token in req.get("args", []):
            if skip_next:
                skip_next = False
                continue
            if isinstance(token, dict) and "input" in token:
                argv.append(str(self.state.input_path(session, token["input"])))
            elif isinstance(token, dict) and "path" in token:
                argv.append(str(Path(str(token["path"])).expanduser()))
            elif isinstance(token, (str, int, float)):
                text = str(token)
                if text in SERVER_OWNED_FLAGS:
                    skip_next = True
                    continue
                argv.append(text)
            else:
                raise ValueError(f"bad argument token: {token!r}")

        if subcommand in MODEL_COMMANDS:
            if not self.state.model:
                raise ValueError("no model configured — set it from the Model button")
            argv += ["--model", self.state.model]
        if subcommand in GEMMA_COMMANDS and self.state.gemma:
            argv += ["--gemma", self.state.gemma]
        if subcommand in QUANTIZE_COMMANDS and req.get("quantize") in {"8", "4", "none"}:
            argv += ["--quantize-on-load", req["quantize"]]
        if subcommand in STEPWISE_COMMANDS and isinstance(req.get("preview"), dict):
            argv += preview_args(req["preview"], self.state.previews_dir(session) / (job_id or uuid.uuid4().hex[:10]))

        output: Path | None = None
        kind = req.get("output", "mp4")
        stamp = datetime.now().strftime("%m%d-%H%M%S")
        stem = safe_name(req.get("take_name") or req.get("task_id") or "take", "take")
        if kind in {"mp4", "dir"}:
            directory = self.state.outputs_dir(session, kind)
            output = self._unique_output(directory, f"{stem}-{stamp}", ".mp4" if kind == "mp4" else "")
            argv += ["--output", str(output)]
        return argv, output, session

    def _unique_output(self, directory: Path, base: str, suffix: str) -> Path:
        """``base`` + suffix, numbered when a file or a queued job already claims it.

        Jobs submitted in the same second (Queue 3 seeds) would otherwise share one
        path and overwrite each other's take.
        """
        with self.cond:
            claimed = {job.get("output") for job in self.jobs.values()}
        candidate, counter = directory / f"{base}{suffix}", 2
        while candidate.exists() or str(candidate) in claimed:
            candidate = directory / f"{base}-{counter}{suffix}"
            counter += 1
        return candidate

    def submit(self, req: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        argv, output, session = self.build_argv(req, job_id)
        preview_dir = self.state.previews_dir(session) / job_id if "--stepwise-image-output-dir" in argv else None
        job = {
            "id": job_id,
            "preview_dir": str(preview_dir) if preview_dir else None,
            "preview_count": 0,
            "session": session,
            "task_id": req.get("task_id"),
            "label": req.get("label") or req.get("task_id"),
            "status": "queued",
            "created": now_iso(),
            "argv": argv,
            "argv_display": "ltx-2-mlx " + " ".join(_quote(a) for a in argv),
            "output": str(output) if output else None,
            "output_kind": req.get("output", "mp4"),
            "params": req.get("params", {}),
            "inputs": input_roles(req.get("args", [])),
            "seed": req.get("seed"),
            "progress": {"phase": "queued", "step": 0, "total": 0, "stage": 0, "stage_key": None},
            "estimate_s": self.state.estimate(session, req.get("params") or {}).get("seconds"),
        }
        self.state.job_queued(job)
        with self.cond:
            self.jobs[job["id"]] = job
            self.pending.append(job["id"])
            self.cond.notify()
        self.emit("queue", self.queue_state())
        return self.summary(job)

    def _keep(self, job: dict[str, Any], output: Path, keep: Callable[[], str]) -> None:
        """Keep what a render made, through helmstudio; what cannot be kept fails the job."""
        try:
            note = keep()
        except Exception as exc:
            job["status"], job["error"] = "failed", f"{output.name} could not be kept: {exc}"
            self.log(job["session"], f"[studio] {output.name} could not be kept: {exc}", kind="err")
            return
        self.log(job["session"], note)

    def cancel(self, job_id: str) -> bool:
        with self.cond:
            if job_id in self.pending:
                self.pending.remove(job_id)
                self.jobs[job_id]["status"] = "cancelled"
                if self.jobs[job_id].get("preview_dir"):
                    shutil.rmtree(self.jobs[job_id]["preview_dir"], ignore_errors=True)
                self.state.job_finished(self.jobs[job_id])
                self.emit(
                    "queue",
                    [self.summary(self.jobs[i]) for i in ([self.current] if self.current else []) + self.pending],
                )
                return True
            if job_id == self.current and self.proc and self.proc.poll() is None:
                self.jobs[job_id]["status"] = "cancelling"
                _stop(self.proc)
                return True
        return False

    # worker ---------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                job_id = self.pending.pop(0)
                self.current = job_id
            try:
                self._run(self.jobs[job_id])
            except Exception as exc:
                job = self.jobs[job_id]
                job["status"], job["error"] = "failed", str(exc)
                if job.get("preview_dir"):
                    shutil.rmtree(job["preview_dir"], ignore_errors=True)
                self.log(job["session"], f"[studio] job failed to start: {exc}", kind="failed")
            finally:
                with self.cond:
                    self.current = None
                    self.proc = None
                self.state.job_finished(self.jobs[job_id])
                self.emit("job", self.summary(self.jobs[job_id]))
                self.emit("queue", self.queue_state())

    def _run(self, job: dict[str, Any]) -> None:
        job["status"], job["started"] = "running", now_iso()
        self.state.job_started(job)
        started = time.monotonic()
        self.emit("job", self.summary(job))
        self.emit("queue", self.queue_state())
        self.log(job["session"], f"$ {job['argv_display']}", kind="cmd")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "0.5"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ltx_pipelines_mlx", *job["argv"]],
            cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        tail: list[str] = []
        watcher_stop = threading.Event()
        watcher = None
        if job.get("preview_dir"):
            watcher = threading.Thread(target=self._watch_previews, args=(job, watcher_stop), daemon=True)
            watcher.start()
        self._pump(job, self.proc.stdout, tail)  # type: ignore[arg-type]
        returncode = self.proc.wait()
        if watcher is not None:
            watcher_stop.set()
            watcher.join(timeout=5)
        job["elapsed"] = round(time.monotonic() - started, 1)
        job["finished"], job["returncode"] = now_iso(), returncode
        cancelled = job["status"] == "cancelling"
        output = Path(job["output"]) if job.get("output") else None
        produced = output is not None and output.exists()
        if cancelled:
            job["status"] = "cancelled"
        elif returncode == 0 and (produced or job["output_kind"] == "none"):
            job["status"] = "done"
        else:
            job["status"] = "failed"
            job["error"] = last_error_line(tail, returncode)
            job["hint"] = hint_for(tail)
        preview_dir = Path(job["preview_dir"]) if job.get("preview_dir") else None
        previews = list_previews(preview_dir) if preview_dir else []
        if job["output_kind"] == "mp4" and produced and output is not None:
            self._keep(job, output, lambda: self.state.take_finished(job, output, previews))
            self.emit("takes", {"session": job["session"]})
        if job["output_kind"] == "dir" and produced and output is not None and job["status"] == "done":
            self._keep(job, output, lambda: self.state.folder_finished(job, output))
        if preview_dir is not None and (job["status"] != "done" or not previews):
            # Nothing owns these previews (no take, or none were written) — don't leave orphans.
            shutil.rmtree(preview_dir, ignore_errors=True)
        state = job["status"]
        self.log(job["session"], f"[studio] {job['label']}: {state} in {job['elapsed']}s", kind=state)

    def _watch_previews(self, job: dict[str, Any], stop: threading.Event) -> None:
        """Push each new stepwise preview to the browser as the pipeline writes it."""
        seen: set[str] = set()
        directory = Path(job["preview_dir"])
        while True:
            finished = stop.wait(0.5)
            for path in list_previews(directory):
                if path.name in seen:
                    continue
                seen.add(path.name)
                info = preview_info(path, self.state.stage)
                job["preview_latest"] = info
                job["preview_count"] = len(seen)
                self.emit("preview", {"id": job["id"], "session": job["session"], "count": len(seen), **info})
            if finished:
                return

    def _pump(self, job: dict[str, Any], stream, tail: list[str]) -> None:
        buf = b""
        last_progress = 0.0
        last_stage = None
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i != -1), default=-1)
                if idx == -1:
                    break
                sep = buf[idx : idx + 1]
                line = buf[:idx].decode("utf-8", "replace")
                buf = buf[idx + 1 :]
                if not line.strip():
                    continue
                replace = sep == b"\r"
                with self.job_lock:
                    changed = self._parse(job, line)
                    progress = dict(job["progress"])
                # Stage changes always go out; step ticks are throttled.
                if changed and (progress.get("stage_key") != last_stage or time.monotonic() - last_progress > 0.25):
                    last_progress, last_stage = time.monotonic(), progress.get("stage_key")
                    self.emit("progress", {"id": job["id"], "progress": progress})
                    self.state.job_progress(job, progress)
                if not replace:
                    tail.append(line)
                    del tail[:-40]
                self.log(job["session"], line, replace)
        if buf.strip():
            line = buf.decode("utf-8", "replace")
            tail.append(line)
            self.log(job["session"], line)

    @staticmethod
    def _parse(job: dict[str, Any], line: str) -> bool:
        progress = job["progress"]
        text = line.strip()

        def advance(stage: str) -> None:
            current = progress.get("stage_key")
            if current is None or STAGES.index(stage) > STAGES.index(current):
                progress["stage_key"] = stage

        if m := REMAINING_RE.search(text):
            seconds = parse_duration(m.group(1))
            if seconds is not None:
                # Wall-clock anchor so the browser can count down between updates.
                progress.update(eta_s=round(seconds), eta_at=time.time())
            return True
        if m := ESTIMATE_RE.search(text):
            progress["stage"] += 1
            progress.update(phase=f"Denoising · stage {progress['stage']}", step=0, total=int(m.group(2)))
            progress.pop("eta_s", None)
            progress.pop("eta_at", None)
            advance("denoise")
            return True
        if m := TQDM_RE.search(text):
            progress.update(step=int(m.group(3)), total=int(m.group(4)))
            return True
        if m := PHASE_RE.match(text):
            progress.update(phase=m.group(1), step=0, total=0)
            for pattern, stage in PHASE_STAGES:
                if pattern.match(m.group(1)):
                    advance(stage)
            return True
        if SAVED_RE.search(text):
            progress.update(phase="Saved", step=0, total=0)
            advance("save")
            return True
        if text.startswith("[auto-duration]") or text.startswith("[official-weights]"):
            progress["note"] = text
            return True
        return False


def preview_args(options: dict[str, Any], directory: Path) -> list[str]:
    """``--stepwise-*`` flags for a live preview, validated into the ranges the CLI accepts."""

    def as_int(key: str, default: int | None, low: int, high: int) -> int | None:
        value = options.get(key, default)
        if value is None or value == "":
            return default
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"preview {key} must be an integer") from exc
        if not low <= number <= high:
            raise ValueError(f"preview {key} must be between {low} and {high}")
        return number

    directory.mkdir(parents=True, exist_ok=True)
    args = [
        "--stepwise-image-output-dir", str(directory),
        "--stepwise-interval", str(as_int("interval", 1, 1, 100)),
        "--stepwise-frames", str(as_int("frames", 8, 1, 32)),
    ]  # fmt: skip
    frame = as_int("frame", None, -512, 512)
    if frame is not None:
        args += ["--stepwise-frame", str(frame)]
    return args


def list_previews(directory: Path) -> list[Path]:
    """Finished preview files in write order (stage, then step). Temp files are skipped."""
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        if m := PREVIEW_NAME.match(path.name):
            found.append((int(m.group(1) or 0), int(m.group(2)), path))
    return [path for _, _, path in sorted(found)]


def preview_info(path: Path, stage_dir: Path) -> dict[str, Any]:
    """A live preview, as the page shows it: served from the stage directory at ``/stage/``."""
    m = PREVIEW_NAME.match(path.name)
    stage, step, total = (int(m.group(1) or 0), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
    rel = _rel(path, stage_dir)
    return {"url": f"/stage/{rel}", "path": rel, "name": path.name, "stage": stage, "step": step, "total": total}


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _quote(arg: str) -> str:
    return arg if re.fullmatch(r"[A-Za-z0-9_./:=@+-]+", arg) else "'" + arg.replace("'", "'\\''") + "'"


def _stop(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# media operations (ffmpeg)
# ---------------------------------------------------------------------------


def _ffmpeg(*argv: str, timeout: float = 120) -> None:
    if not which("ffmpeg"):
        raise ValueError("ffmpeg is not on PATH")
    try:
        result = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", *argv], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffmpeg timed out after {timeout:.0f}s") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "ffmpeg failed")


def grab_frame(src: Path, dst: Path, position: str) -> None:
    """A video's first frame, or its last, as an image."""
    if position == "first":
        _ffmpeg("-i", str(src), "-frames:v", "1", str(dst))
    else:
        _ffmpeg("-sseof", "-0.1", "-i", str(src), "-frames:v", "1", "-update", "1", str(dst))


def grab_audio(src: Path, dst: Path) -> None:
    """A video's audio track, as stereo audio."""
    _ffmpeg("-i", str(src), "-vn", "-ac", "2", str(dst))


def receive(body: BinaryIO, length: int, dst: Path) -> None:
    """Write ``length`` bytes of a request body to ``dst``, a megabyte at a time."""
    with open(dst, "wb") as f:
        while length > 0:
            chunk = body.read(min(1 << 20, length))
            if not chunk:
                break
            f.write(chunk)
            length -= len(chunk)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "ltx-studio"
    state: State
    runner: Runner
    allowed_hosts: ClassVar[set[str]] = set()

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet access log
        return

    # plumbing -------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"error": message}, code)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _session(self, params: dict[str, Any]) -> str:
        return self.state.session_name(params.get("session"))

    def _refused(self) -> bool:
        """Answer and return True when the request fails ``request_guard``."""
        refusal = request_guard(self.command, self.path, self.headers, self.allowed_hosts)
        if refusal is None:
            return False
        code, message = refusal
        # "message" as well as "error": a refusal on the /helm/ proxy is read by
        # helmstudio's SDK, which takes "error" for a code and shows "message".
        # With only "error" it has nothing to say and its caller falls back to
        # "that change was not made".
        self._json({"error": message, "message": message}, code)
        return True

    def _proxied(self) -> bool:
        """Serve a request under /helm/ through helmstudio's proxy: the page's SDK files, API calls and assets."""
        if not urlparse(self.path).path.startswith(PREFIX):
            return False
        return self.state.proxy.handle_http(self)

    # routes ---------------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_PUT(self) -> None:
        if self._refused() or self._proxied():
            return None
        return self._error("not found", 404)

    def do_PATCH(self) -> None:
        self.do_PUT()

    def do_DELETE(self) -> None:
        self.do_PUT()

    def do_GET(self) -> None:
        if self._refused() or self._proxied():
            return None
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path.startswith("/stage/"):
                return self._file(self.state.resolve_stage_path(path[len("/stage/") :]))
            if path == "/api/timeline":
                return self._json(self.state.list_timeline(self._session(query)))
            if path == "/api/events":
                return self._events()
            if path == "/api/config":
                return self._json({
                    "model": self.state.model_info(), "sessions": self.state.sessions(),
                    "active": self.state.active, "ffmpeg": bool(which("ffmpeg")),
                    "preferences": self.state.preferences(),
                })  # fmt: skip
            if path == "/api/sessions":
                return self._json({"sessions": self.state.sessions(), "active": self.state.active})
            if path == "/api/inputs":
                return self._json(self.state.list_inputs(self._session(query)))
            if path == "/api/takes":
                return self._json(self.state.list_takes(self._session(query)))
            if path == "/api/queue":
                return self._json(self.runner.queue_state())
            if path == "/api/terminal":
                return self._json({"job": self.state.terminal_job(self._session(query))})
            return self._error("not found", 404)
        except ValueError as exc:
            return self._error(str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:  # never drop the connection without a response
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:
        if self._refused() or self._proxied():
            return None
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        try:
            if path == "/api/upload":
                return self._upload(query)
            body = self._body()
            session = self._session(body)
            if path == "/api/render":
                jobs = [self.runner.submit(req) for req in body.get("jobs", [body])]
                return self._json({"jobs": jobs})
            if path == "/api/cancel":
                return self._json({"ok": self.runner.cancel(str(body.get("id")))})
            if path == "/api/estimate":
                return self._json(self.state.estimate(session, body.get("params") or {}))
            if path == "/api/takes/star":
                return self._json(self.state.set_star(session, str(body.get("name")), bool(body.get("starred"))))
            if path == "/api/model/check":
                return self._json(self.state.model_info())
            if path == "/api/model":
                model = str(body.get("model", "")).strip()
                if model and not Path(model).expanduser().exists() and "/" not in model:
                    return self._error("model path does not exist")
                with self.state.lock:
                    self.state.model = model
                    self.state.gemma = str(body.get("gemma", "")).strip() or None
                return self._json(self.state.model_info())
            if path == "/api/session/activate":
                return self._json(self.state.activate(session))
            if path == "/api/session/save":
                self.state.save_settings(session, body.get("settings", {}))
                return self._json({"ok": True})
            if path == "/api/preferences":
                changes = body.get("changes")
                if not isinstance(changes, dict):
                    raise ValueError("changes must be an object")
                self.state.save_preferences(changes)
                return self._json({"ok": True})
            if path == "/api/session/duplicate":
                return self._json(self.state.duplicate_session(session, str(body.get("new_name", ""))))
            if path == "/api/session/delete":
                return self._json(self.state.delete_session(session))
            if path == "/api/inputs/delete":
                self.state.delete_input(session, str(body.get("name")))
                return self._json({"ok": True})
            if path == "/api/takes/delete":
                self.state.delete_take(session, str(body.get("name")))
                return self._json({"ok": True})
            if path == "/api/frame":
                return self._json(self.state.frame(session, str(body["take"]), body.get("position", "last")))
            if path == "/api/audio":
                return self._json(self.state.audio(session, str(body["take"])))
            if path == "/api/use-video":
                kind = body.get("kind", "outputs")
                return self._json(self.state.use_video(session, kind, str(body.get("name") or body["take"])))
            if path == "/api/timeline/delete":
                self.state.delete_timeline(session, str(body.get("name")))
                return self._json({"ok": True})
            return self._error("not found", 404)
        except (ValueError, KeyError) as exc:
            return self._error(str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:  # never drop the connection without a response
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # handlers -------------------------------------------------------------
    def _static(self, rel: str) -> None:
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in target.parents:
            return self._error("not found", 404)
        return self._file(target)

    def _file(self, target: Path) -> None:
        if not target.is_file():
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        return self._send(200, target.read_bytes(), ctype)

    def _upload(self, query: dict[str, str]) -> None:
        original = Path(unquote(self.headers.get("X-Filename") or "upload")).name
        length = int(self.headers.get("Content-Length") or 0)
        return self._json(self.state.save_upload(self._session(query), original, self.rfile, length))

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.runner.subscribe()
        try:
            self.wfile.write(f"data: {json.dumps({'type': 'queue', 'data': self.runner.queue_state()})}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    message = q.get(timeout=15)
                    self.wfile.write(f"data: {message}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.runner.unsubscribe(q)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Browsers abort media range requests all the time; that is not an error.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    parser = argparse.ArgumentParser(description="ltx studio web UI")
    parser.add_argument("--model", default=os.environ.get("LTX_MODEL", ""), help="model dir or HF repo (env LTX_MODEL)")
    parser.add_argument("--gemma", default=os.environ.get("LTX_GEMMA"), help="Gemma 3 repo for LTX-2.3 packs / enhance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8720)
    parser.add_argument(
        "--allow-host", default="", help="extra Host names to accept, comma-separated (IPs and localhost always are)"
    )
    args = parser.parse_args()

    try:
        client, proxy = connect()
    except UnavailableError as exc:
        sys.exit(f"ltx studio: {exc}")
    state = State(args.model, args.gemma, client, proxy)
    Handler.state = state
    Handler.runner = Runner(state)
    Handler.allowed_hosts = {h.strip().lower() for h in [args.host, *args.allow_host.split(",")] if h.strip()}
    httpd = StudioServer((args.host, args.port), Handler)
    print(f"ltx studio on http://{args.host}:{args.port}  (model: {args.model or 'not set'})", flush=True)
    print(f"helmstudio keeps everything ltx studio keeps; scratch in {state.stage}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: no authentication — anyone who can reach this port can run jobs", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()


if __name__ == "__main__":
    main()
