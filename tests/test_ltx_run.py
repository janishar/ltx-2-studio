"""Unit tests for scripts/ltx_run.py (mode launcher). Stdlib only, no MLX or weights."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ltx_run", REPO_ROOT / "scripts" / "ltx_run.py")
assert _spec is not None and _spec.loader is not None
ltx_run = importlib.util.module_from_spec(_spec)
sys.modules["ltx_run"] = ltx_run
_spec.loader.exec_module(ltx_run)


@pytest.fixture
def official_dir(tmp_path: Path) -> Path:
    root = tmp_path / "official"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / ltx_run.OFFICIAL_DISTILLED).write_bytes(b"")
    return root


@pytest.fixture
def media(tmp_path: Path) -> dict[str, Path]:
    files = {name: tmp_path / name for name in ("a.png", "b.png", "clip.mp4", "song.wav")}
    for path in files.values():
        path.write_bytes(b"")
    return files


def _plan(argv: list[str]):
    args = ltx_run.build_parser().parse_args(argv)
    args.passthrough = []
    width, height = ltx_run.SIZE_PRESETS[args.size]
    args.width, args.height = args.width or width, args.height or height
    args.frames = args.frames or ltx_run.seconds_to_frames(args.seconds, args.fps)
    return ltx_run.build_plan(args)


def _value(argv: list[str], flag: str, count: int = 1) -> list[str]:
    i = argv.index(flag)
    return argv[i + 1 : i + 1 + count]


def test_seconds_to_frames_is_8k_plus_1():
    assert ltx_run.seconds_to_frames(2.0, 24) == 49
    assert ltx_run.seconds_to_frames(5.0, 24) == 121
    assert ltx_run.seconds_to_frames(0.1, 24) == 9
    assert ltx_run.seconds_to_latent(1.5, 24) == 4


def test_parse_anchor():
    assert ltx_run._parse_anchor("img.png@1.5") == ("img.png", 1.5, 1.0)
    assert ltx_run._parse_anchor("dir@x/img.png@2@0.5") == ("dir@x/img.png", 2.0, 0.5)
    with pytest.raises(SystemExit):
        ltx_run._parse_anchor("img.png")


def test_inspect_official_and_pack(official_dir: Path, tmp_path: Path):
    info = ltx_run.inspect_model(str(official_dir))
    assert (info.local, info.has_distilled, info.has_dev, info.is_25) == (True, True, False, True)
    assert ltx_run.mode_availability(info, "t2v") is None
    assert "dev" in ltx_run.mode_availability(info, "retake")
    assert "dev" in ltx_run.mode_availability(info, "t2v", "two-stage")
    assert "LTX-2.3" in ltx_run.mode_availability(info, "v2v")

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "transformer-dev.safetensors").write_bytes(b"")
    (pack / "transformer-distilled.safetensors").write_bytes(b"")
    (pack / "embedded_config.json").write_text(json.dumps({"transformer": {}}))
    info = ltx_run.inspect_model(str(pack))
    assert (info.has_distilled, info.has_dev, info.is_25) == (True, True, False)
    assert ltx_run.mode_availability(info, "v2v") is None

    # Dev without the distilled LoRA: stage 2 of two-stage, hq, a2v and keyframe can't run.
    (official_dir / "diffusion_models" / ltx_run.OFFICIAL_DEV).write_bytes(b"")
    info = ltx_run.inspect_model(str(official_dir))
    assert info.has_dev and not info.has_distilled_lora
    for mode, pipeline in (("t2v", "two-stage"), ("t2v", "hq"), ("a2v", "distilled"), ("keyframe", "distilled")):
        assert "distilled LoRA" in ltx_run.mode_availability(info, mode, pipeline)
    for mode, pipeline in (("t2v", "one-stage"), ("retake", "distilled"), ("extend", "distilled")):
        assert ltx_run.mode_availability(info, mode, pipeline) is None
    (official_dir / "loras").mkdir()
    (official_dir / "loras" / ltx_run.OFFICIAL_DISTILLED_LORA).write_bytes(b"")
    info = ltx_run.inspect_model(str(official_dir))
    assert info.has_distilled_lora and ltx_run.mode_availability(info, "t2v", "two-stage") is None

    remote = ltx_run.inspect_model("some-org/some-repo")
    assert not remote.local and ltx_run.mode_availability(remote, "a2v") is None


def test_generate_modes(official_dir: Path, media: dict[str, Path]):
    common = ["--model", str(official_dir), "-p", "x", "-o", "out.mp4"]

    t2v = _plan(["t2v", *common, "--seconds", "3", "--size", "720p"]).argv
    assert t2v[0] == "generate" and "--distilled" in t2v
    assert _value(t2v, "--frames") == ["73"]
    assert _value(t2v, "--width") == ["1280"] and _value(t2v, "--height") == ["704"]
    assert _value(t2v, "--quantize-on-load") == ["8"]

    i2v = _plan(["i2v", *common, "--image", str(media["a.png"])]).argv
    assert _value(i2v, "--image", 3) == [str(media["a.png"]), "0", "1.0"]

    flf = _plan(["flf2v", *common, "--first", str(media["a.png"]), "--last", str(media["b.png"])]).argv
    images = [flf[i + 1 : i + 4] for i, token in enumerate(flf) if token == "--image"]
    assert images == [[str(media["a.png"]), "0", "1.0"], [str(media["b.png"]), "48", "1.0"]]

    anchors = _plan(["anchors", *common, "--anchor", f"{media['a.png']}@1@0.5", "--anchor", f"{media['b.png']}@9"]).argv
    frames = [anchors[i + 2] for i, token in enumerate(anchors) if token == "--image"]
    assert frames == ["24", "48"]  # 9 s is clamped to the last frame

    story = _plan(["story", *common, "--beat", "one", "--beat", "two"]).argv
    assert [story[i + 1] for i, token in enumerate(story) if token == "--segment"] == ["one", "two"]

    two_stage = _plan(["t2v", *common, "--pipeline", "hq"]).argv
    assert "--two-stages-hq" in two_stage and "--distilled" not in two_stage


def test_auto_duration_omits_frames(official_dir: Path):
    argv = _plan(["t2v", "--model", str(official_dir), "-p", "x", "-o", "o.mp4", "--auto-duration"]).argv
    assert "--frames" not in argv


def test_other_modes(official_dir: Path, media: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    common = ["--model", str(official_dir), "-p", "x", "-o", "out.mp4"]
    monkeypatch.setattr(ltx_run, "probe_video", lambda path: (24.0, 4.0))

    retake = _plan(["retake", *common, "--video", str(media["clip.mp4"]), "--from", "1", "--to", "2", "--keep-audio"])
    assert _value(retake.argv, "--start") == ["3"] and _value(retake.argv, "--end") == ["6"]
    assert "--no-regen-audio" in retake.argv

    extend = _plan(
        ["extend", *common, "--video", str(media["clip.mp4"]), "--add-seconds", "1", "--direction", "before"]
    )
    assert _value(extend.argv, "--extend-frames") == ["3"] and _value(extend.argv, "--direction") == ["before"]

    a2v = _plan(["a2v", *common, "--audio", str(media["song.wav"]), "--audio-start", "2.5"]).argv
    assert a2v[0] == "a2v" and _value(a2v, "--audio-start") == ["2.5"]

    keyframe = _plan(["keyframe", *common, "--first", str(media["a.png"]), "--last", str(media["b.png"])]).argv
    assert _value(keyframe, "--start") == [str(media["a.png"])]

    v2v = _plan(["v2v", *common, "--lora", "org/lora", "--control", str(media["clip.mp4"])]).argv
    assert v2v[0] == "ic-lora" and _value(v2v, "--lora", 2) == ["org/lora", "1.0"]

    lipdub = _plan(["lipdub", *common, "--video", str(media["clip.mp4"]), "--lora", "org/lipdub"]).argv
    assert lipdub[0] == "lipdub" and "--frames" not in lipdub


def test_missing_inputs_fail(official_dir: Path, tmp_path: Path):
    with pytest.raises(SystemExit):
        _plan(["i2v", "--model", str(official_dir), "-p", "x", "--image", str(tmp_path / "nope.png")])
    with pytest.raises(SystemExit):
        _plan(["v2v", "--model", str(official_dir), "-p", "x", "--lora", "org/lora"])


def test_main_dry_run_passthrough_and_gating(official_dir: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        ltx_run.main(["t2v", "--model", str(official_dir), "-p", "x", "-o", "o.mp4", "--dry-run", "--", "--no-audio"])
        == 0
    )
    assert capsys.readouterr().out.strip().endswith("--no-audio")

    with pytest.raises(SystemExit, match="dev transformer"):
        ltx_run.main(["keyframe", "--model", str(official_dir), "-p", "x", "--first", "a", "--last", "b"])
    with pytest.raises(SystemExit, match="8k"):
        ltx_run.main(["t2v", "--model", str(official_dir), "-p", "x", "--frames", "50", "--dry-run"])
