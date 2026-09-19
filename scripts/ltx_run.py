#!/usr/bin/env python3
"""One launcher for every way this repo can generate video.

Thin front end over ``ltx-2-mlx``: each mode is named by its inputs, takes
seconds and paths instead of latent-frame indices, picks sensible defaults,
and prints the exact ``ltx-2-mlx`` command it runs.

Modes:
    t2v       text -> video                    (generate)
    i2v       image -> video                   (generate --image)
    flf2v     first + last frame -> video      (generate, two image anchors)
    anchors   several images at chosen times   (generate, N image anchors)
    story     timed prompt beats -> video      (generate --segment, Prompt Relay)
    a2v       audio (+ image) -> video         (a2v; dev model)
    retake    regenerate a time range of video (retake; dev model)
    extend    add seconds before/after a video (extend; dev model)
    keyframe  interpolate between two images   (keyframe; dev model)
    v2v       control video + IC-LoRA -> video (ic-lora)
    hdr       HDR IC-LoRA video                (hdr-ic-lora)
    lipdub    re-sync a video to its audio     (lipdub)
    modes     show which modes the model supports
    demo      run every supported mode, chaining outputs as inputs

Examples:
    export LTX_MODEL=/path/to/Lightricks-LTX-2.5
    uv run python scripts/ltx_run.py t2v -p "a fox in the snow" --seconds 3
    uv run python scripts/ltx_run.py i2v -p "she turns and smiles" --image face.jpg
    uv run python scripts/ltx_run.py flf2v -p "day turns to night" --first day.png --last night.png
    uv run python scripts/ltx_run.py story -p "cinematic kitchen" --beat "chopping onions" --beat "serving the dish"
    uv run python scripts/ltx_run.py modes
    uv run python scripts/ltx_run.py demo --size small

Anything after ``--`` is passed through to ``ltx-2-mlx`` unchanged.
Stdlib only: generation runs in a child process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"
MODEL_ENV = "LTX_MODEL"

#: WIDTHxHEIGHT presets. Two-stage pipelines generate at half size first, so
#: every preset is a multiple of 64 on both axes (otherwise output dims snap down).
SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "small": (704, 448),
    "square": (768, 768),
    "portrait": (448, 704),
    "sd": (896, 512),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

OFFICIAL_DISTILLED = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
OFFICIAL_DEV = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
#: The official AV video VAE, which carries the LTX-2.5 diffusion video decoder.
OFFICIAL_VIDEO_VAE_AV = "ltx-2.5-video-vae-bf16.safetensors"

#: Generation pipelines of ``generate`` and whether they need the dev transformer.
GENERATE_PIPELINES: dict[str, tuple[str, bool]] = {
    "distilled": ("--distilled", False),
    "two-stage": ("--two-stage", True),
    "hq": ("--two-stages-hq", True),
    "one-stage": ("--one-stage", True),
}

#: Mode-specific options, so a namespace built for one mode can drive any other (demo).
MODE_OPTION_DEFAULTS: dict[str, object] = {
    "image": None,
    "first": None,
    "last": None,
    "anchor": None,
    "beat": None,
    "audio": None,
    "audio_start": 0.0,
    "video": None,
    "from_": 0.0,
    "to": None,
    "keep_audio": False,
    "add_seconds": 2.0,
    "direction": "after",
    "lora": None,
    "lora_strength": 1.0,
    "control": None,
}

DEV_MODES = {"a2v", "retake", "extend", "keyframe"}
IC_LORA_MODES = {"v2v", "hdr", "lipdub"}
GENERATE_MODES = {"t2v", "i2v", "flf2v", "anchors", "story"}


# ---------------------------------------------------------------------------
# Model inspection (filesystem only, no MLX)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """What a ``--model`` value can run, as far as the filesystem tells."""

    model: str
    local: bool
    has_distilled: bool
    has_dev: bool
    is_25: bool
    #: The model carries the LTX-2.5 diffusion video decoder
    #: (``vae_decoder_av.safetensors``, or the official AV video VAE).
    has_diffusion_decoder: bool = False

    def missing_dev_reason(self) -> str:
        return "needs the dev transformer (not found in model)"

    def missing_diffusion_decoder_reason(self) -> str:
        return (
            "needs the LTX-2.5 diffusion video decoder, which this model does not have "
            f"(expected {OFFICIAL_VIDEO_VAE_AV} or vae_decoder_av.safetensors)"
        )


def _walk_names(root: Path) -> set[str]:
    names: set[str] = set()
    for _dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        names.update(filenames)
    return names


def inspect_model(model: str) -> ModelInfo:
    """Classify ``model`` without loading anything.

    A HuggingFace repo id cannot be inspected offline, so every mode is
    assumed available and ltx-2-mlx reports anything missing.
    """
    path = Path(model).expanduser()
    if not path.is_dir():
        return ModelInfo(model, local=False, has_distilled=True, has_dev=True, is_25=False, has_diffusion_decoder=True)

    names = _walk_names(path)
    official = OFFICIAL_DISTILLED in names or OFFICIAL_DEV in names
    has_distilled = OFFICIAL_DISTILLED in names or any(
        n == "transformer.safetensors" or n.startswith("transformer-distilled") for n in names
    )
    has_dev = OFFICIAL_DEV in names or "transformer-dev.safetensors" in names

    is_25 = official
    config = path / "embedded_config.json"
    if not is_25 and config.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            is_25 = json.loads(config.read_text()).get("transformer", {}).get("ff_bias") is False
    has_diffusion_decoder = OFFICIAL_VIDEO_VAE_AV in names or "vae_decoder_av.safetensors" in names
    return ModelInfo(
        model,
        local=True,
        has_distilled=has_distilled,
        has_dev=has_dev,
        is_25=is_25,
        has_diffusion_decoder=has_diffusion_decoder,
    )


def mode_availability(
    info: ModelInfo, mode: str, pipeline: str = "distilled", video_decoder: str = "conv"
) -> str | None:
    """Return why ``mode`` cannot run on ``info``, or ``None`` if it can."""
    if mode in GENERATE_MODES:
        if video_decoder == "diffusion" and not info.has_diffusion_decoder:
            return info.missing_diffusion_decoder_reason()
        _, needs_dev = GENERATE_PIPELINES[pipeline]
        if needs_dev and not info.has_dev:
            return f"--pipeline {pipeline} " + info.missing_dev_reason()
        if not needs_dev and not info.has_distilled:
            return "needs the distilled transformer (not found in model)"
        return None
    if mode in DEV_MODES and not info.has_dev:
        return info.missing_dev_reason()
    if mode in IC_LORA_MODES and info.is_25:
        return "IC-LoRA modes run on LTX-2.3 packs only in ltx-2-studio"
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Nearest valid pixel frame count (8k + 1, at least 9) for a duration."""
    return max(1, round(seconds * fps / 8)) * 8 + 1


def seconds_to_latent(seconds: float, fps: float) -> int:
    """Latent frame index for a time in the video (8 pixel frames per latent frame)."""
    return max(0, round(seconds * fps / 8))


def probe_video(path: Path) -> tuple[float, float]:
    """Return ``(fps, duration_seconds)`` of a video using ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data = json.loads(out)
    num, den = data["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den), float(data["format"]["duration"])


def default_output(out_dir: Path, mode: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return out_dir / f"{mode}-{stamp}-s{seed}.mp4"


def _require_file(path: str | None, flag: str) -> str:
    if not path:
        raise SystemExit(f"error: {flag} is required")
    if not Path(path).expanduser().exists():
        raise SystemExit(f"error: {flag} file not found: {path}")
    return str(Path(path).expanduser())


def _parse_anchor(spec: str) -> tuple[str, float, float]:
    """``IMAGE@SECONDS[@STRENGTH]`` -> (image, seconds, strength)."""
    parts = spec.rsplit("@", 2)
    if len(parts) < 2:
        raise SystemExit(f"error: --anchor expects IMAGE@SECONDS[@STRENGTH], got {spec!r}")
    if len(parts) == 3:
        image, seconds, strength = parts
    else:
        (image, seconds), strength = parts, "1.0"
    return image, float(seconds), float(strength)


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """A resolved ltx-2-mlx invocation."""

    mode: str
    argv: list[str]
    output: Path
    notes: list[str] = field(default_factory=list)


def _common(args: argparse.Namespace, subcommand: str, output: Path) -> list[str]:
    argv = [
        subcommand,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--output",
        str(output),
        "--seed",
        str(args.seed),
    ]
    argv += ["--quantize-on-load", args.quantize]
    if args.low_ram:
        argv.append("--low-ram")
    # Only `generate` has --video-decoder; the other subcommands would reject it.
    video_decoder = getattr(args, "video_decoder", "conv")
    if subcommand == "generate" and video_decoder != "conv":
        argv += ["--video-decoder", video_decoder]
    return argv


def _size_frames(args: argparse.Namespace, *, frames: bool = True) -> list[str]:
    width, height = args.width, args.height
    argv = ["--width", str(width), "--height", str(height), "--frame-rate", str(args.fps)]
    if frames and not args.auto_duration:
        argv += ["--frames", str(args.frames)]
    return argv


def build_plan(args: argparse.Namespace) -> Plan:
    """Translate a parsed mode invocation into an ltx-2-mlx command."""
    mode = args.mode
    output = Path(args.output) if args.output else default_output(args.out_dir, mode, args.seed)
    notes: list[str] = []

    if mode in GENERATE_MODES:
        argv = _common(args, "generate", output) + [GENERATE_PIPELINES[args.pipeline][0]] + _size_frames(args)
        if args.no_audio:
            argv.append("--no-audio")
        last_frame = (args.frames - 1) if not args.auto_duration else None
        if mode == "i2v":
            argv += ["--image", _require_file(args.image, "--image"), "0", str(args.strength)]
        elif mode == "flf2v":
            if last_frame is None:
                raise SystemExit("error: flf2v needs a fixed length; drop --auto-duration")
            argv += ["--image", _require_file(args.first, "--first"), "0", str(args.strength)]
            argv += ["--image", _require_file(args.last, "--last"), str(last_frame), str(args.strength)]
        elif mode == "anchors":
            if not args.anchor:
                raise SystemExit("error: anchors needs at least one --anchor IMAGE@SECONDS[@STRENGTH]")
            for spec in args.anchor:
                image, seconds, strength = _parse_anchor(spec)
                frame = min(round(seconds * args.fps), args.frames - 1)
                argv += ["--image", _require_file(image, "--anchor"), str(frame), str(strength)]
        elif mode == "story":
            if not args.beat:
                raise SystemExit("error: story needs at least one --beat TEXT")
            for beat in args.beat:
                argv += ["--segment", beat]
            if args.image:
                argv += ["--image", _require_file(args.image, "--image"), "0", str(args.strength)]
    elif mode == "a2v":
        argv = _common(args, "a2v", output) + _size_frames(args, frames=False)
        argv += ["--audio", _require_file(args.audio, "--audio")]
        if args.audio_start:
            argv += ["--audio-start", str(args.audio_start)]
        if not args.auto_duration:
            argv += ["--frames", str(args.frames)]
        if args.image:
            argv += ["--image", _require_file(args.image, "--image"), "0", str(args.strength)]
    elif mode in ("retake", "extend"):
        video = _require_file(args.video, "--video")
        fps, duration = probe_video(Path(video))
        argv = _common(args, mode, output) + ["--video", video]
        if mode == "retake":
            end_seconds = duration if args.to is None else args.to
            start, end = seconds_to_latent(args.from_, fps), max(seconds_to_latent(end_seconds, fps), 1)
            if end <= start:
                end = start + 1
            argv += ["--start", str(start), "--end", str(end)]
            if args.keep_audio:
                argv.append("--no-regen-audio")
            notes.append(
                f"retake {args.from_:.2f}s-{end_seconds:.2f}s -> latent frames [{start}, {end}) at {fps:g} fps"
            )
        else:
            latent = max(1, math.ceil(args.add_seconds * fps / 8))
            argv += ["--extend-frames", str(latent), "--direction", args.direction]
            notes.append(f"extend {args.direction} by {latent} latent frames (~{latent * 8 / fps:.2f}s at {fps:g} fps)")
    elif mode == "keyframe":
        argv = _common(args, "keyframe", output) + _size_frames(args)
        argv += ["--start", _require_file(args.first, "--first"), "--end", _require_file(args.last, "--last")]
    elif mode in ("v2v", "hdr"):
        subcommand = "ic-lora" if mode == "v2v" else "hdr-ic-lora"
        argv = _common(args, subcommand, output) + _size_frames(args)
        if not args.lora:
            raise SystemExit(f"error: {mode} needs --lora PATH_OR_REPO")
        argv += ["--lora", args.lora, str(args.lora_strength)]
        if args.control:
            argv += ["--video-conditioning", _require_file(args.control, "--control"), str(args.strength)]
        elif mode == "v2v":
            raise SystemExit("error: v2v needs --control VIDEO (depth / canny / pose / motion tracks)")
        if args.image:
            argv += ["--image", _require_file(args.image, "--image"), "0", str(args.strength)]
    elif mode == "lipdub":
        argv = _common(args, "lipdub", output) + ["--width", str(args.width), "--height", str(args.height)]
        if not args.lora:
            raise SystemExit("error: lipdub needs --lora PATH_OR_REPO")
        argv += [
            "--reference-video",
            _require_file(args.video, "--video"),
            "--lora",
            args.lora,
            str(args.lora_strength),
        ]
    else:
        raise SystemExit(f"error: unknown mode {mode!r}")

    return Plan(mode, argv + list(args.passthrough), output, notes)


def run_plan(plan: Plan, *, dry_run: bool) -> tuple[bool, float]:
    """Run (or print) a plan. Returns ``(succeeded, seconds)``."""
    command = [sys.executable, "-m", "ltx_pipelines_mlx", *plan.argv]
    for note in plan.notes:
        print(f"[ltx-run] {note}")
    print("[ltx-run] ltx-2-mlx " + shlex.join(plan.argv), flush=True)
    if dry_run:
        return True, 0.0
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    ok = result.returncode == 0 and plan.output.exists()
    print(f"[ltx-run] {plan.mode}: {'done' if ok else 'FAILED'} in {elapsed:.1f}s -> {plan.output}", flush=True)
    return ok, elapsed


# ---------------------------------------------------------------------------
# modes / demo
# ---------------------------------------------------------------------------

MODE_SUMMARY: dict[str, str] = {
    "t2v": "text -> video",
    "i2v": "image -> video",
    "flf2v": "first + last frame -> video",
    "anchors": "images at chosen times -> video",
    "story": "timed prompt beats -> video",
    "a2v": "audio (+ image) -> video",
    "retake": "regenerate a time range of a video",
    "extend": "add seconds before/after a video",
    "keyframe": "interpolate between two images (dev + CFG)",
    "v2v": "control video + IC-LoRA -> video",
    "hdr": "HDR IC-LoRA -> video",
    "lipdub": "lip-sync a video to its audio",
}


def cmd_modes(args: argparse.Namespace) -> int:
    info = inspect_model(args.model)
    kind = ("LTX-2.5" if info.is_25 else "LTX-2.3 or other") if info.local else "HuggingFace repo (not inspected)"
    print(f"model: {args.model}  [{kind}; distilled={info.has_distilled} dev={info.has_dev}]\n")
    for mode, summary in MODE_SUMMARY.items():
        reason = mode_availability(info, mode)
        status = "ready" if reason is None else f"unavailable: {reason}"
        print(f"  {mode:9s} {summary:44s} {status}")
    for pipeline in ("two-stage", "hq", "one-stage"):
        reason = mode_availability(info, "t2v", pipeline)
        print(f"  {'t2v':9s} {'--pipeline ' + pipeline:44s} {'ready' if reason is None else 'unavailable: ' + reason}")
    if info.local and not info.has_dev:
        print(f"\nDev-model modes need {OFFICIAL_DEV} (official) or transformer-dev.safetensors (mlx-forge pack).")
    if info.is_25:
        print("IC-LoRA modes (v2v, hdr, lipdub) need an LTX-2.3 pack plus the matching Lightricks IC-LoRA.")
    return 0


def _ffmpeg(*argv: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *argv], check=True)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run every mode the model supports, chaining generated media as inputs."""
    info = inspect_model(args.model)
    run_dir = args.out_dir / f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results: list[tuple[str, str, float, Path | None]] = []
    base = run_dir / "t2v.mp4"
    first, last, audio = run_dir / "first.png", run_dir / "last.png", run_dir / "audio.wav"

    def attempt(mode: str, **overrides: object) -> bool:
        ns = argparse.Namespace(
            **{**MODE_OPTION_DEFAULTS, **vars(args), "mode": mode, "output": str(run_dir / f"{mode}.mp4"), **overrides}
        )
        reason = mode_availability(info, mode, ns.pipeline)
        if reason is not None:
            results.append((mode, f"skipped: {reason}", 0.0, None))
            print(f"[ltx-run] skip {mode}: {reason}")
            return False
        plan = build_plan(ns)
        ok, elapsed = run_plan(plan, dry_run=args.dry_run)
        results.append((mode, "ok" if ok else "FAILED", elapsed, plan.output))
        return ok

    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    base_ok = attempt("t2v", prompt=prompt)
    if base_ok and not args.dry_run:
        fps, duration = probe_video(base)
        _ffmpeg("-i", str(base), "-frames:v", "1", str(first))
        _ffmpeg("-sseof", "-0.1", "-i", str(base), "-frames:v", "1", "-update", "1", str(last))
        _ffmpeg("-i", str(base), "-vn", "-ac", "2", str(audio))
    elif args.dry_run:
        for placeholder in (base, first, last, audio):
            placeholder.touch()
        duration = args.frames / args.fps

    if base_ok:
        attempt("i2v", image=str(first))
        attempt("flf2v", first=str(first), last=str(last))
        attempt("anchors", anchor=[f"{first}@0", f"{last}@{max(duration - 0.1, 0.5):.2f}@0.8"])
        attempt("story", beat=[args.beats[0], args.beats[1]] if len(args.beats) >= 2 else args.beats)
        attempt("a2v", audio=str(audio))
        attempt("retake", video=str(base), from_=duration * 0.4, to=duration * 0.8)
        attempt("extend", video=str(base), add_seconds=1.0)
        attempt("keyframe", first=str(first), last=str(last))
        for pipeline in ("two-stage", "hq"):
            reason = mode_availability(info, "t2v", pipeline)
            if reason is None:
                attempt("t2v", pipeline=pipeline, output=str(run_dir / f"t2v-{pipeline}.mp4"))
            else:
                results.append((f"t2v --pipeline {pipeline}", f"skipped: {reason}", 0.0, None))
    for mode in sorted(IC_LORA_MODES):
        results.append((mode, "skipped: needs --lora and a control/reference video (run the mode directly)", 0.0, None))

    print(f"\n[ltx-run] demo summary ({run_dir})")
    for mode, status, elapsed, output in results:
        timing = f"{elapsed:7.1f}s" if elapsed else " " * 8
        print(f"  {mode:26s} {timing} {status}{'  ' + str(output) if output and status == 'ok' else ''}")
    return 0 if all(not s.startswith("FAILED") for _, s, _, _ in results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("common")
    g.add_argument("--model", "-m", default=os.environ.get(MODEL_ENV), help=f"model dir or HF repo (env {MODEL_ENV})")
    g.add_argument("--output", "-o", help="output .mp4 (default: outputs/<mode>-<time>-s<seed>.mp4)")
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory for default outputs")
    g.add_argument("--size", choices=sorted(SIZE_PRESETS), default="small", help="WIDTHxHEIGHT preset (default: small)")
    g.add_argument("--width", "-W", type=int, help="override preset width")
    g.add_argument("--height", "-H", type=int, help="override preset height")
    g.add_argument("--seconds", type=float, default=2.0, help="clip length (default: 2.0)")
    g.add_argument("--frames", "-f", type=int, help="exact pixel frame count (8k+1); overrides --seconds")
    g.add_argument("--auto-duration", action="store_true", help="let the LTX-2.5 DurationHead pick the length")
    g.add_argument("--fps", type=float, default=24.0, help="frame rate (default: 24)")
    g.add_argument("--seed", "-s", type=int, default=42)
    g.add_argument("--quantize", choices=["8", "4", "none"], default="8", help="official-weights quantization")
    g.add_argument("--pipeline", choices=sorted(GENERATE_PIPELINES), default="distilled", help="generate pipeline")
    g.add_argument("--strength", type=float, default=1.0, help="image / control conditioning strength")
    g.add_argument("--low-ram", action="store_true", help="block streaming (converted packs only)")
    g.add_argument("--no-audio", action="store_true", help="skip audio decode/mux (generate modes)")
    g.add_argument(
        "--video-decoder",
        choices=["conv", "diffusion"],
        default="conv",
        help="video VAE decoder for generate modes: conv (default) or the LTX-2.5 diffusion decoder",
    )
    g.add_argument("--dry-run", action="store_true", help="print the ltx-2-mlx command without running it")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltx_run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="MODE")

    def mode(name: str, needs_prompt: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=MODE_SUMMARY.get(name, name))
        if needs_prompt:
            p.add_argument("--prompt", "-p", required=True)
        _add_common(p)
        return p

    mode("t2v")
    mode("i2v").add_argument("--image", "-i", required=True, help="start image")
    p = mode("flf2v")
    p.add_argument("--first", required=True, help="first frame image")
    p.add_argument("--last", required=True, help="last frame image")
    mode("anchors").add_argument("--anchor", action="append", help="IMAGE@SECONDS[@STRENGTH], repeatable")
    p = mode("story")
    p.add_argument("--beat", action="append", help="local prompt for the next slice of time, repeatable")
    p.add_argument("--image", "-i", help="optional start image")
    p = mode("a2v")
    p.add_argument("--audio", "-a", required=True, help="audio file driving the video")
    p.add_argument("--audio-start", type=float, default=0.0, help="seconds into the audio to start")
    p.add_argument("--image", "-i", help="optional start image")
    p = mode("retake")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--from", dest="from_", type=float, default=0.0, help="start second (default: 0)")
    p.add_argument("--to", type=float, help="end second (default: end of video)")
    p.add_argument("--keep-audio", action="store_true", help="keep the original audio")
    p = mode("extend")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--add-seconds", type=float, default=2.0, help="seconds to add (default: 2)")
    p.add_argument("--direction", choices=["after", "before"], default="after")
    p = mode("keyframe")
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    for name in ("v2v", "hdr"):
        p = mode(name)
        p.add_argument("--lora", help="IC-LoRA path or HF repo")
        p.add_argument("--lora-strength", type=float, default=1.0)
        p.add_argument("--control", help="control video (depth / canny / pose / tracks / SDR source)")
        p.add_argument("--image", "-i", help="optional start image")
    p = mode("lipdub")
    p.add_argument("--video", "-v", required=True, help="reference video whose audio drives the lips")
    p.add_argument("--lora", help="LipDub IC-LoRA path or HF repo")
    p.add_argument("--lora-strength", type=float, default=1.0)

    p = sub.add_parser("modes", help="show which modes the model supports")
    p.add_argument("--model", "-m", default=os.environ.get(MODEL_ENV))
    p = mode("demo")
    p.set_defaults(prompt=None)
    for action in p._actions:
        if action.dest == "prompt":
            action.required = False
            action.default = "A golden retriever runs along a sunlit beach, waves breaking behind it."
    p.add_argument(
        "--beats",
        nargs="+",
        default=["the dog runs toward the camera", "the dog stops and shakes off water"],
        help="story beats used by the demo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1 :]
    args = build_parser().parse_args(argv)
    args.passthrough = passthrough

    if not args.model:
        raise SystemExit(f"error: pass --model or set {MODEL_ENV}")
    if args.mode == "modes":
        return cmd_modes(args)

    preset_w, preset_h = SIZE_PRESETS[args.size]
    args.width = args.width or preset_w
    args.height = args.height or preset_h
    args.frames = args.frames or seconds_to_frames(args.seconds, args.fps)
    if (args.frames - 1) % 8:
        raise SystemExit(f"error: --frames must be 8k+1 (e.g. 49, 97, 121), got {args.frames}")

    if args.mode == "demo":
        return cmd_demo(args)

    info = inspect_model(args.model)
    reason = mode_availability(info, args.mode, args.pipeline, args.video_decoder)
    if reason is not None:
        if not args.dry_run:
            raise SystemExit(f"error: {args.mode} {reason} (use --dry-run to see the command anyway)")
        print(f"[ltx-run] warning: {args.mode} {reason}")
    ok, _ = run_plan(build_plan(args), dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
