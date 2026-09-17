"""Unit tests for ltx studio's server, web/server.py, and what it keeps through helmstudio.

Stdlib only: no MLX, no HTTP server, no weights. helmstudio is played by the
in-memory stand-in in ``tests/web_studio.py``.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.web_studio import REPO_ROOT, apply_merge_patch, fake_helmstudio, runner_for, server


@pytest.fixture
def platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return fake_helmstudio(tmp_path, monkeypatch)


@pytest.fixture
def state(platform):
    return server.State("/models/ltx-2.5", None, platform.client(), proxy=None)


@pytest.fixture
def runner(state):
    return runner_for(state)


def upload(state, name: str, data: bytes, session: str = "session-1") -> dict[str, Any]:
    return state.save_upload(session, name, io.BytesIO(data), len(data))


def test_safe_name():
    assert server.safe_name("my shot/../x") == "my-shot-..-x"
    assert server.safe_name("  ") == "untitled"


def test_the_server_keeps_nothing_without_helmstudio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HELM_API", raising=False)
    monkeypatch.setattr(server, "from_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["server.py", "--port", "0"])
    with pytest.raises(SystemExit, match="helm dev"):
        server.main()
    monkeypatch.setattr(server, "from_env", None)  # helm-runtime-sdk not installed
    with pytest.raises(SystemExit, match="helm-runtime-sdk is not installed"):
        server.main()
    assert not (REPO_ROOT / "web" / "sessions").exists()


def test_build_argv_generate(platform, state, runner):
    state.activate("session-1")
    upload(state, "face.png", b"png bytes")
    argv, output, _ = runner.build_argv({
        "session": "session-1", "subcommand": "generate", "quantize": "8", "take_name": "shot 1",
        "args": ["--prompt", "hi", "--distilled", "--image", {"input": "face.png"}, "0", "1.0",
                 "--output", "/tmp/elsewhere.mp4", "--model", "other/model"],
    })  # fmt: skip
    assert argv[0] == "generate"
    image = Path(argv[argv.index("--image") + 1])
    assert image.read_bytes() == b"png bytes" and platform.stage in image.parents
    assert "/tmp/elsewhere.mp4" not in argv and "other/model" not in argv
    assert argv[argv.index("--model") + 1] == "/models/ltx-2.5"
    assert argv[argv.index("--quantize-on-load") + 1] == "8"
    assert output is not None and output.parent == platform.stage / "takes" and output.name.startswith("shot-1-")
    assert argv[argv.index("--output") + 1] == str(output)


def test_build_argv_rejects_bad_requests(state, runner):
    with pytest.raises(ValueError, match="unsupported command"):
        runner.build_argv({"subcommand": "rm", "args": []})
    with pytest.raises(ValueError, match="input not found"):
        runner.build_argv({"subcommand": "generate", "args": [{"input": "missing.png"}]})
    with pytest.raises(ValueError, match="bad argument"):
        runner.build_argv({"subcommand": "generate", "args": [["nested"]]})


def test_build_argv_tools_have_no_model_or_output(state, runner):
    argv, output, _ = runner.build_argv({"subcommand": "enhance", "output": "none", "args": ["--prompt", "x"]})
    assert "--model" not in argv and "--output" not in argv and output is None
    argv, output, _ = runner.build_argv({"subcommand": "slice", "output": "dir", "args": [{"path": "~/clips"}]})
    assert "--model" not in argv and output is not None and str(Path("~/clips").expanduser()) in argv
    # A folder a training tool writes goes to the studio's data directory, where the next tool reads it by path.
    assert output.parent == state.data / "outputs" / state.session(state.active)["id"]


def test_progress_parsing():
    job = {"progress": {"phase": "", "step": 0, "total": 0, "stage": 0}}
    assert server.Runner._parse(job, "[Loading transformer (transformer.safetensors)] ...")
    assert job["progress"]["phase"].startswith("Loading transformer")
    assert server.Runner._parse(job, "[estimate] denoising (ancestral): 8 steps x 1 passes over 539 video tokens")
    assert job["progress"]["stage"] == 1 and job["progress"]["total"] == 8
    assert server.Runner._parse(job, "Denoising (ancestral):  50%|█████     | 4/8 [00:04<00:04,  1.0s/it]")
    assert (job["progress"]["step"], job["progress"]["total"]) == (4, 8)
    assert not server.Runner._parse(job, "some unrelated line")


def test_ffprobe_missing_file_is_empty(tmp_path: Path):
    assert server.ffprobe(tmp_path / "nope.mp4") == {}


def test_stage_paths_cannot_escape(platform, state):
    root = platform.stage.resolve()
    assert state.resolve_stage_path("previews/job") == root / "previews" / "job"
    assert state.resolve_stage_path("") == root
    for bad in ("../secrets", "previews/../../etc/passwd", "%2e%2e/etc"):
        with pytest.raises(ValueError):
            state.resolve_stage_path(bad)


def test_preview_args_and_validation(platform, state, runner, tmp_path: Path):
    argv, _, _ = runner.build_argv(
        {
            "subcommand": "generate",
            "args": ["--stepwise-image-output-dir", "/tmp/elsewhere"],
            "preview": {"interval": 2, "frames": 3, "frame": -1},
        },
        job_id="job123",
    )
    directory = platform.stage / "previews" / "job123"
    assert argv[argv.index("--stepwise-image-output-dir") + 1] == str(directory) and directory.is_dir()
    assert "/tmp/elsewhere" not in argv
    assert argv[argv.index("--stepwise-interval") + 1] == "2"
    assert argv[argv.index("--stepwise-frames") + 1] == "3"
    assert argv[argv.index("--stepwise-frame") + 1] == "-1"

    argv, _, _ = runner.build_argv({"subcommand": "generate", "args": [], "preview": {"frame": None}})
    assert "--stepwise-frame" not in argv and argv[argv.index("--stepwise-frames") + 1] == "8"
    argv, _, _ = runner.build_argv({"subcommand": "enhance", "output": "none", "args": [], "preview": {}})
    assert "--stepwise-image-output-dir" not in argv

    for bad in ({"interval": 0}, {"frames": 99}, {"frames": "many"}):
        with pytest.raises(ValueError):
            server.preview_args(bad, tmp_path / "p")


def test_list_previews_orders_by_stage_then_step(platform):
    directory = platform.stage / "previews" / "job"
    directory.mkdir(parents=True)
    for name in ("seed_1_s2_step001of003.webp", "seed_1_s1_step010of008.webp", "seed_1_s1_step002of008.webp",
                 "seed_1_s1_step003of008.webp.tmp", "notes.txt"):  # fmt: skip
        (directory / name).write_bytes(b"")
    names = [p.name for p in server.list_previews(directory)]
    assert names == ["seed_1_s1_step002of008.webp", "seed_1_s1_step010of008.webp", "seed_1_s2_step001of003.webp"]
    info = server.preview_info(directory / "seed_1_s2_step001of003.webp", platform.stage)
    assert (info["stage"], info["step"], info["total"]) == (2, 1, 3)
    assert info["url"] == "/stage/previews/job/seed_1_s2_step001of003.webp"
    single = server.preview_info(directory / "seed_-5_step004of008.webp", platform.stage)
    assert (single["stage"], single["step"], single["total"]) == (0, 4, 8)


def test_request_guard_blocks_rebinding_and_cross_site():
    allowed = {"127.0.0.1", "studio.lan"}
    guard = server.request_guard
    json_post = {"Host": "127.0.0.1:8720", "Origin": "http://127.0.0.1:8720", "Content-Type": "application/json"}
    assert guard("POST", "/api/render", json_post, allowed) is None
    assert guard("GET", "/api/config", {"Host": "localhost:8720"}, allowed) is None
    assert guard("GET", "/api/config", {"Host": "[::1]:8720"}, allowed) is None
    assert guard("GET", "/api/config", {"Host": "studio.lan:8720"}, allowed) is None
    assert guard("GET", "/stage/x", {"Host": "evil.example:8720"}, allowed)[0] == 403
    assert guard("GET", "/", {}, allowed)[0] == 403
    assert guard("POST", "/api/render", {**json_post, "Origin": "http://evil.example"}, allowed)[0] == 403
    assert guard("POST", "/api/render", {**json_post, "Origin": "null"}, allowed)[0] == 403
    no_origin = {"Host": "127.0.0.1:8720", "Content-Type": "application/json"}
    assert guard("POST", "/api/render", no_origin, allowed) is None
    assert guard("POST", "/api/render", {**no_origin, "Sec-Fetch-Site": "cross-site"}, allowed)[0] == 403
    # A no-cors form/fetch post can only send "simple" content types.
    assert guard("POST", "/api/session/delete", {**json_post, "Content-Type": "text/plain"}, allowed)[0] == 415
    assert guard("POST", "/api/session/delete", {"Host": "127.0.0.1:8720"}, allowed)[0] == 415
    assert guard("POST", "/api/upload?session=s", {"Host": "127.0.0.1:8720"}, allowed)[0] == 400
    assert guard("POST", "/api/upload?session=s", {"Host": "127.0.0.1:8720", "X-Filename": "a.png"}, allowed) is None
    # helmstudio's proxy: the page's own merge patches, still only from its own origin.
    patch = {"Host": "127.0.0.1:8720", "Content-Type": "application/merge-patch+json"}
    assert guard("PATCH", "/helm/api/v1/gallery/items/x", patch, allowed) is None
    assert guard("PATCH", "/helm/api/v1/gallery/items/x", {**patch, "Origin": "http://evil.example"}, allowed)[0] == 403


def test_progress_eta_and_stages():
    job = {"progress": {"phase": "", "step": 0, "total": 0, "stage": 0, "stage_key": None}}
    parse = server.Runner._parse
    assert parse(job, "[Loading text encoder (Gemma)] ...") and job["progress"]["stage_key"] == "encode"
    assert (
        parse(job, "[Loading transformer (transformer-dev.safetensors)] ...") and job["progress"]["stage_key"] == "load"
    )
    assert parse(job, "[estimate] denoising: 30 steps x 2 passes over 1650 video + 200 audio tokens = 60 forwards")
    assert job["progress"]["stage_key"] == "denoise"
    assert parse(job, "[estimate] denoising: ~1 min 30 s remaining (1.5 s/forward) (refined)")
    assert job["progress"]["eta_s"] == 90 and job["progress"]["eta_at"] > 0
    # Stage 2 reloads the transformer: the stepper never moves backwards, and the old ETA is dropped.
    assert parse(job, "[Loading transformer (transformer-distilled.safetensors)] ...")
    assert job["progress"]["stage_key"] == "denoise"
    assert parse(job, "[estimate] denoising: 3 steps x 1 passes over 6600 video + 200 audio tokens = 3 forwards")
    assert job["progress"]["stage"] == 2 and "eta_s" not in job["progress"]
    assert parse(job, "[Loading decoders (VAE + audio + vocoder)] ...") and job["progress"]["stage_key"] == "decode"
    assert parse(job, "[Decoding video + audio + muxing] ...") and job["progress"]["stage_key"] == "decode"
    assert parse(job, "Saved to: /tmp/x.mp4") and job["progress"]["stage_key"] == "save"


def test_parse_duration():
    assert server.parse_duration("5 s") == 5
    assert server.parse_duration("1 min 30 s") == 90
    assert server.parse_duration("1 h 30 min") == 5400
    assert server.parse_duration("soon") is None


def test_failure_hints():
    watchdog = ["Loading...", "libc++abi: [METAL] Command buffer execution failed: Impacting Interactivity (0000000e)"]
    assert "watchdog" in server.hint_for(watchdog)
    oom = [
        "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
    ]
    assert "memory" in server.hint_for(oom)
    assert "DurationHead" in server.hint_for(["ValueError: ... Pass num_frames explicitly."])
    assert "dev transformer" in server.hint_for(["FileNotFoundError: --dev-transformer 'x' not found in model dir: /m"])
    # Normal loading lines that merely mention the dev transformer don't trigger that hint.
    assert server.hint_for(["[Loading transformer (transformer-dev.safetensors)] ...", "KeyError: 'x'"]) == ""
    tail = ["[Loading transformer] ...", "Traceback (most recent call last):", "ValueError: bad size", "  at end"]
    assert server.last_error_line(tail, 1) == "ValueError: bad size"
    assert server.last_error_line([], 3) == "exit code 3"


def _take(elapsed: float, width: int, frames: int, task: str = "t2v") -> tuple[float, dict]:
    params = {"taskId": task, "common": {"width": width, "height": 448, "frames": frames, "quantize": "8"},
              "values": {task: {"pipeline": "distilled", "prompt_unrelated": "x"}}}  # fmt: skip
    return elapsed, params


def test_estimate_from_takes():
    params = {"taskId": "t2v", "common": {"width": 704, "height": 448, "frames": 49, "quantize": "8"},
              "values": {"t2v": {"pipeline": "distilled"}}}  # fmt: skip
    assert server.estimate_from([], params) == {}
    takes = [_take(100, 704, 97)]
    assert server.estimate_from(takes, params) == {"seconds": 51, "samples": 1, "exact": False}
    takes += [_take(40, 704, 49), _take(60, 704, 49), _take(999, 704, 49, task="i2v"), (None, params)]
    assert server.estimate_from(takes, params) == {"seconds": 50, "samples": 2, "exact": True}
    assert server.estimate_from(takes, {"common": {}}) == {}


def test_setup_checks(tmp_path: Path):
    pack = tmp_path / "pack"
    pack.mkdir()
    for name in ("transformer-distilled.safetensors", "vae_decoder.safetensors"):
        (pack / name).write_bytes(b"")
    checks = {c["label"]: c for c in server.setup_checks(server.LTX_RUN.inspect_model(str(pack)))}
    assert checks["transformer"]["level"] == "ok" and checks["VAE decoder"]["level"] == "ok"
    assert checks["upscaler"]["level"] == "warn"
    assert server.setup_checks(server.LTX_RUN.inspect_model("/does/not/exist"))[0]["level"] == "bad"
    assert server.setup_checks(server.LTX_RUN.inspect_model("Lightricks/LTX-2.5"))[0]["level"] == "info"
    assert server.setup_checks(None)[0]["level"] == "bad"
    assert server.checks_status([{"level": "ok"}, {"level": "info"}]) == "warn"


def test_jobs_submitted_together_get_distinct_outputs(platform, state, runner):
    """Queue 3 seeds submits in the same second; each take needs its own file."""
    req = {"session": "session-1", "subcommand": "generate", "task_id": "t2v", "args": ["--prompt", "x"]}
    outputs = [runner.submit({**req, "seed": seed})["output"] for seed in (1, 2, 3)]
    assert len(set(outputs)) == 3
    assert all(Path(o).parent == platform.stage / "takes" for o in outputs)
    job = runner.jobs[runner.pending[0]]
    assert runner.summary(job)["progress"]["stage_key"] is None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ({}, {"taskId": "t2v", "common": {"prompt": "a", "seed": 1}, "values": {}}),
        (
            {"taskId": "t2v", "common": {"prompt": "a", "seed": 1, "takeName": "x"}},
            {"taskId": "i2v", "common": {"prompt": "a"}},
        ),
        (
            {"values": {"i2v": {"loras": [{"path": "p", "strength": 1}], "image": "a.png"}}},
            {"values": {"i2v": {"loras": []}}},
        ),
        ({"common": {"preview": {"enabled": True}}}, {"common": "replaced"}),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_merge_patch_turns_old_into_new(old, new):
    assert apply_merge_patch(old, server.merge_patch(old, new)) == new
    assert server.merge_patch(new, new) == {}


def test_a_state_document_holds_no_nulls():
    settings = {
        "common": {"seed": None, "prompt": "a"},
        "values": {"i2v": {"image": None}},
        "rows": [None, {"a": None}],
    }
    stored = server.without_nulls(settings)
    assert stored == {"common": {"prompt": "a"}, "values": {"i2v": {}}, "rows": [None, {"a": None}]}
    assert apply_merge_patch({}, settings) == stored


def test_input_roles_name_each_input_by_the_flag_before_it():
    args = ["--prompt", "x", "--image", {"input": "a.png"}, "0", "1.0", "--image", {"input": "b.png"}, "-1", "1.0",
            "--video-conditioning", {"input": "c.mp4"}, "0.5", "--start", {"input": "d.png"}]  # fmt: skip
    assert server.input_roles(args) == [("a.png", "image"), ("b.png", "image"), ("c.mp4", "video_conditioning"),
                                        ("d.png", "start")]  # fmt: skip
    assert server.input_roles([{"input": "e.png"}]) == [("e.png", "input")]


def test_percent_weighs_stages_as_the_page_does():
    assert server.percent({"stage_key": None}) == 0
    assert server.percent({"stage_key": "load"}) == 8
    assert server.percent({"stage_key": "denoise", "stage": 1, "step": 4, "total": 8}) == 45
    assert server.percent({"stage_key": "denoise", "stage": 2, "step": 3, "total": 3}) == 88
    assert server.percent({"stage_key": "save"}) == 98


def test_sessions_and_their_settings_are_helmstudios(platform, state):
    assert state.sessions() == [] and state.active == "session-1"
    assert state.activate("session-1") == {"name": "session-1", "settings": {}}
    assert state.sessions() == ["session-1"] and not (platform.data / "sessions").exists()

    settings = {"taskId": "t2v", "common": {"prompt": "a", "seed": 1, "takeName": ""}, "values": {}}
    state.save_settings("session-1", settings)
    state.save_settings("session-1", {**settings, "common": {"prompt": "b", "seed": 1}})
    state.save_settings("session-1", {**settings, "common": {"prompt": "b", "seed": 1}})  # unchanged: nothing to write
    assert [update["state"]["settings"] for update in platform.updates] == [
        settings, {"common": {"prompt": "b", "takeName": None}}]  # fmt: skip
    assert state.activate("session-1")["settings"] == {
        "taskId": "t2v",
        "common": {"prompt": "b", "seed": 1},
        "values": {},
    }

    # Another writer changed the session since it was read: read it again and apply the change to what is there.
    row = platform.session_rows[state.session("session-1")["id"]]
    row["etag"], row["state"]["settings"]["taskId"] = "written elsewhere", "i2v"
    state.save_settings("session-1", {**settings, "taskId": "a2v"})
    assert row["state"]["settings"] == {**settings, "taskId": "a2v"}

    # The session opened last is the one a new start opens.
    state.activate("other")
    state.activate("session-1")
    assert server.State("/models/ltx-2.5", None, platform.client(), proxy=None).active == "session-1"


def test_an_upload_is_a_pinned_asset_adopted_from_the_stage(platform, state):
    state.activate("session-1")
    first = upload(state, "my face.png", b"png bytes")
    second = upload(state, "my face.png", b"png bytes")
    assert first["name"] == "my-face.png" and second["name"].startswith("my-face-") and second["kind"] == "image"
    (asset,) = platform.asset_rows.values()  # the same bytes are one asset
    assert asset["pinned"] and asset["kind"] == "image"
    assert not [path for path in state.stage.rglob("*") if path.is_file()]  # adopted: nothing left in the stage
    assert {item["url"] for item in state.list_inputs("session-1")} == {f"/helm/api/v1/assets/{asset['id']}"}

    state.delete_input("session-1", first["name"])
    assert [item["name"] for item in state.list_inputs("session-1")] == [second["name"]] and asset["pinned"]


def test_extracting_again_replaces_the_input_and_an_upload_does_not(state):
    state.activate("session-1")
    state.put_input(
        "session-1", "take-last.png", {"asset_id": "A", "kind": "image", "probe": {"fps": 25}}, replace=True
    )
    state.put_input("session-1", "take-last.png", {"asset_id": "B", "kind": "image", "probe": {}}, replace=True)
    assert state.inputs("session-1") == {"take-last.png": {"asset_id": "B", "kind": "image", "probe": {}}}
    assert state.put_input("session-1", "take-last.png", {"asset_id": "C", "kind": "image"}) != "take-last.png"


def test_duplicating_a_session_copies_its_settings_and_inputs(state):
    state.activate("one")
    state.save_settings("one", {"taskId": "i2v"})
    upload(state, "face.png", b"png bytes", session="one")
    assert state.duplicate_session("one", "two") == {"name": "two", "settings": {"taskId": "i2v"}}
    assert [item["name"] for item in state.list_inputs("two")] == ["face.png"] and state.active == "two"
    with pytest.raises(ValueError, match="already exists"):
        state.duplicate_session("one", "two")
    assert state.delete_session("two")["name"] == "one" and state.sessions() == ["one"]


def test_a_take_is_a_gallery_item_with_its_session_and_inputs(platform, state, runner, monkeypatch):
    probe = {"width": 704, "height": 448, "duration": 2.0, "fps": 24, "frames": 49, "has_audio": True}
    monkeypatch.setattr(server, "ffprobe", lambda path: probe)
    state.activate("session-1")
    name = upload(state, "face.png", b"png bytes")["name"]
    params = {"taskId": "i2v", "common": {"prompt": "a cat", "width": 704, "height": 448, "frames": 49}, "values": {}}
    request = {"session": "session-1", "subcommand": "generate", "task_id": "i2v", "label": "Image → Video", "seed": 7,
               "args": ["--prompt", "a cat", "--image", {"input": name}, "0", "1.0"], "params": params}  # fmt: skip
    job = runner.jobs[runner.submit(request)["id"]]
    output = Path(job["output"])
    output.write_bytes(b"mp4 bytes")
    job["elapsed"] = 12.0
    preview = platform.stage / "previews" / job["id"] / "seed_7_step001of008.webp"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"webp")
    note = state.take_finished(job, output, [preview])

    (item,) = platform.items.values()
    image = next(asset for asset in platform.asset_rows.values() if asset["kind"] == "image")
    assert item["session_id"] == state.session("session-1")["id"] and item["id"] in note
    assert item["inputs"] == [{"asset_id": image["id"], "role": "image"}]
    assert (
        item["params"]["prompt"] == "a cat" and item["params"]["name"] == output.name and "argv" not in item["params"]
    )
    assert not output.exists() and not platform.asset_rows[item["asset_id"]]["pinned"]

    (take,) = state.list_takes("session-1")
    assert take["name"] == output.name and take["url"] == f"/helm/api/v1/assets/{item['asset_id']}"
    assert (take["params"], take["probe"], take["seed"]) == (params, probe, 7)
    assert take["previews"] == [f"previews/{job['id']}/seed_7_step001of008.webp"]
    preview.unlink()  # helmstudio clears its stage directory when it stops the studio
    assert state.list_takes("session-1")[0]["previews"] == []
    assert state.estimate("session-1", params) == {"seconds": 12, "samples": 1, "exact": True}
    assert state.set_star("session-1", output.name, True)["starred"] and state.list_takes("session-1")[0]["starred"]

    # Used as an input, the take's asset is pinned: the session now depends on it.
    used = state.use_video("session-1", "outputs", output.name)["name"]
    assert (
        state.inputs("session-1")[used]["asset_id"] == item["asset_id"]
        and platform.asset_rows[item["asset_id"]]["pinned"]
    )

    state.duplicate_session("session-1", "copy")
    assert state.list_takes("copy") == []  # a take stays with the session that made it
    state.delete_take("session-1", output.name)
    assert state.list_takes("session-1") == []
    with pytest.raises(ValueError, match="take not found"):
        state.set_star("session-1", output.name, False)


def test_a_render_is_a_job_and_its_log_is_the_terminal(platform, state, runner, monkeypatch):
    monkeypatch.setattr(server.HelmJob, "INTERVAL_S", 0)
    monkeypatch.setattr(server.HelmJob, "LOG_INTERVAL_S", 3600)  # the test sends the log itself
    state.activate("session-1")
    request = {"session": "session-1", "subcommand": "enhance", "output": "none", "args": ["--prompt", "x"]}
    job = runner.jobs[runner.submit(request)["id"]]
    (reported,) = platform.job_rows.values()
    assert (reported["state"], reported["subject_kind"], reported["subject_id"]) == (
        "queued", "session", state.session("session-1")["id"])  # fmt: skip
    assert runner.summary(job)["helm_job"] == reported["id"]  # the page points helm-terminal at it
    assert state.terminal_job("session-1") is None  # a queued render has no log to show yet

    job["status"] = "running"
    state.job_started(job)
    runner.log("session-1", "$ ltx-2-mlx enhance", kind="cmd")
    runner.log("session-1", "Denoising:  50%", replace=True)
    runner.log("session-1", "Denoising:  75%", replace=True)  # an update not sent yet gives way to the next
    state.render_report(job["id"]).flush()
    runner.log("session-1", "Denoising: 100%", replace=True)
    runner.log("session-1", "Denoising: 100%")  # the bar closes: its final line draws over its last update
    runner.log("session-1", "[Encoding prompt] ...")
    runner.log("another session", "not this render's")
    state.render_report(job["id"]).flush()
    assert platform.job_logs[reported["id"]] == [
        "\x1b[36m$ ltx-2-mlx enhance\x1b[0m",  # helm-terminal draws it in the log accent
        "Denoising:  75%\r",  # and rewrites it in place when the next line comes
        "Denoising: 100%",
        "[Encoding prompt] ...",
    ]
    state.job_progress(job, {"stage_key": "decode"})
    assert (reported["state"], reported["progress_num"], reported["progress_den"]) == ("running", 88, 100)
    assert state.terminal_job("session-1") == reported["id"]
    assert state.terminal_job("another session") is None

    runner.log("session-1", "[studio] enhance: done in 1.0s", kind="done")
    job["status"] = "done"
    state.job_finished(job)  # sends what is left, then ends the job
    assert (reported["state"], reported["progress_num"]) == ("succeeded", 100)
    assert platform.job_logs[reported["id"]][-1] == "\x1b[36m[studio] enhance: done in 1.0s\x1b[0m"
    assert state.terminal_job("session-1") == reported["id"]  # a finished render's log stays on screen

    failed = runner.jobs[runner.submit(request)["id"]]
    failed.update(status="failed", error="Out of memory")
    state.job_finished(failed)
    assert platform.job_rows[max(platform.job_rows)]["last_error"] == {
        "code": "render_failed",
        "message": "Out of memory",
    }

    queued = runner.submit(request)
    assert runner.cancel(queued["id"]) and platform.job_rows[max(platform.job_rows)]["state"] == "cancelled"
    assert state.terminal_job("session-1") == max(platform.job_rows)  # the latest render, cancelled or not


def test_a_render_log_reaches_the_terminal_while_the_render_is_quiet(platform, state, runner, monkeypatch):
    """A line printed before a minute of denoising is sent on a timer, not when the next line comes."""
    monkeypatch.setattr(server.HelmJob, "LOG_INTERVAL_S", 0.01)
    state.activate("session-1")
    job = runner.jobs[runner.submit({"session": "session-1", "subcommand": "enhance", "output": "none"})["id"]]
    job["status"] = "running"
    state.job_started(job)
    runner.log("session-1", "Loading the transformer")
    log = platform.job_logs[job["helm_job"]]
    deadline = time.monotonic() + 5
    while not log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log == ["Loading the transformer"]
    job["status"] = "done"
    state.job_finished(job)
    assert platform.job_rows[job["helm_job"]]["state"] == "succeeded"


def test_helmstudio_can_cancel_a_render(platform, state, runner):
    state.activate("session-1")
    queued = runner.submit({"session": "session-1", "subcommand": "enhance", "output": "none", "args": []})
    (reported,) = platform.job_rows.values()
    state.follow_cancellations(runner.cancel)
    platform.events.put(
        SimpleNamespace(id="1", name="job", json=lambda: {"job": {**reported, "cancel_requested_at": "now"}})
    )
    deadline = time.monotonic() + 5
    while runner.jobs[queued["id"]]["status"] != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.jobs[queued["id"]]["status"] == "cancelled" and reported["state"] == "cancelled"


def test_a_training_tools_folder_is_kept_as_assets_and_a_record(platform, state, runner):
    state.activate("session-1")
    request = {"session": "session-1", "subcommand": "slice", "task_id": "slice", "label": "Slice Clips",
               "output": "dir", "args": []}  # fmt: skip
    job = runner.jobs[runner.submit(request)["id"]]
    folder = Path(job["output"])
    (folder / "clips").mkdir(parents=True)
    (folder / "clips" / "clip-001.mp4").write_bytes(b"clip")
    (folder / "clips" / "clip-001.txt").write_text("a caption")

    note = state.folder_finished(job, folder)
    (record,) = platform.records["folders"]
    document = record["doc"]
    assert document["session_id"] == state.session("session-1")["id"] and record["id"] in note
    assert document["path"] == f"outputs/{state.session('session-1')['id']}/{folder.name}"
    assert {platform.asset_rows[asset]["kind"] for asset in document["files"].values()} == {"video", "other"}
    assert set(document["files"]) == {"clips/clip-001.mp4", "clips/clip-001.txt"}
    # Adopted where they are: the next tool still reads the folder by path.
    assert (folder / "clips" / "clip-001.txt").read_text() == "a caption"


def test_the_pages_preferences_are_a_kv_document(platform, state):
    assert state.preferences() == {}
    state.save_preferences({"side": "timeline"})
    state.save_preferences({"terminalHeight": 320, "notify": True})
    state.save_preferences({"notify": None})
    assert state.preferences() == {"side": "timeline", "terminalHeight": 320}
    assert platform.kv == {server.PREFERENCES: {"side": "timeline", "terminalHeight": 320}}


class FakePipeline:
    """``python -m ltx_pipelines_mlx``, as far as a render needs: it writes its ``--output`` and exits 0."""

    def __init__(self, argv: list[str], **_: Any) -> None:
        output = Path(argv[argv.index("--output") + 1])
        if argv[3] == "slice":
            (output / "source").mkdir(parents=True)
            (output / "source" / "source_000.mp4").write_bytes(b"clip")
        else:
            output.write_bytes(b"take")
        self.stdout = io.BytesIO(f"Saved to: {output}\n".encode())
        self.pid = 0

    def wait(self) -> int:
        return 0

    def poll(self) -> int:
        return 0


def test_a_finished_render_keeps_what_it_made_through_helmstudio(platform, state, runner, monkeypatch):
    real_popen = server.subprocess.Popen

    def popen(argv: list[str], **kwargs: Any) -> Any:
        return FakePipeline(argv) if argv[1:3] == ["-m", "ltx_pipelines_mlx"] else real_popen(argv, **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server, "ffprobe", lambda path: {})
    state.activate("session-1")
    for request in ({"subcommand": "generate", "task_id": "t2v", "args": ["--prompt", "x"]},
                    {"subcommand": "slice", "task_id": "slice", "output": "dir", "args": [{"path": "/clips"}]}):  # fmt: skip
        job = runner.jobs[runner.submit({"session": "session-1", "label": request["task_id"], **request})["id"]]
        runner._run(job)
        assert job["status"] == "done", job.get("error")
        state.job_finished(job)

    (take,) = platform.items.values()
    assert take["session_id"] == state.session("session-1")["id"] and not list((platform.stage / "takes").iterdir())
    (folder,) = platform.records["folders"]
    assert list(folder["doc"]["files"]) == ["source/source_000.mp4"]
    assert [reported["state"] for reported in platform.job_rows.values()] == ["succeeded", "succeeded"]
