"""ltx studio task catalog (web/static/tasks.js), run through Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TASKS_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "tasks.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

MODEL_25 = {
    "configured": True,
    "local": True,
    "is_25": True,
    "has_distilled": True,
    "has_dev": True,
    "has_diffusion_decoder": True,
}
MODEL_23 = {**MODEL_25, "is_25": False, "has_diffusion_decoder": False}
MODEL_HF = {**MODEL_25, "local": False, "is_25": False}
#: An LTX-2.5 pack predating the diffusion decoder (no vae_decoder_av weights).
MODEL_25_NO_AV = {**MODEL_25, "has_diffusion_decoder": False}


def _node(script: str) -> object:
    """Run ``script`` with tasks.js loaded as ``t`` and return what it prints as JSON."""
    code = f"const t = require({json.dumps(str(TASKS_JS))});\n{script}"
    out = subprocess.run(["node", "-e", code], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_generated_keyframes_flag_on_every_generate_task() -> None:
    result = _node(
        """const ctx = {frames: 49, fps: 24, width: 704, height: 448, seed: 1, inputMeta: () => ({})};
        const out = {};
        for (const task of t.LTX_TASKS.filter((x) => x.cmd === "generate")) {
          const fields = task.fields.map((f) => f.key);
          const args = task.build({generatedKeyframes: 3, beats: [], anchors: []}, ctx);
          const off = task.build({generatedKeyframes: 0, beats: [], anchors: []}, ctx);
          out[task.id] = [fields.includes("generatedKeyframes"),
                          args.join(" ").includes("--num-generated-keyframes 3"),
                          off.includes("--num-generated-keyframes")];
        }
        console.log(JSON.stringify(out));"""
    )
    assert result and all(v == [True, True, False] for v in result.values()), result


def test_generated_keyframes_requirements() -> None:
    cases = _node(
        f"""const r = (v, model, ctx) => t.generateRequires(v, model, ctx);
        const m25 = {json.dumps(MODEL_25)}, m23 = {json.dumps(MODEL_23)}, hf = {json.dumps(MODEL_HF)};
        console.log(JSON.stringify({{
          off23: r({{generatedKeyframes: 0}}, m23, {{frames: 49}}),
          blank23: r({{generatedKeyframes: ""}}, m23, {{frames: 49}}),
          ok25: r({{generatedKeyframes: 3}}, m25, {{frames: 49}}),
          on23: r({{generatedKeyframes: 3}}, m23, {{frames: 49}}),
          hf: r({{generatedKeyframes: 3}}, hf, {{frames: 49}}),
          tooShort: r({{generatedKeyframes: 8}}, m25, {{frames: 9}}),
          autoDuration: r({{generatedKeyframes: 8}}, m25, {{frames: 9, autoDuration: true}}),
          fractional: r({{generatedKeyframes: 1.5}}, m25, {{frames: 49}}),
          negative: r({{generatedKeyframes: -1}}, m25, {{frames: 49}}),
          noCtx: r({{generatedKeyframes: 2}}, m25),
        }}));"""
    )
    assert cases["off23"] is None and cases["blank23"] is None
    assert cases["ok25"] is None and cases["hf"] is None and cases["autoDuration"] is None and cases["noCtx"] is None
    assert "LTX-2.5" in cases["on23"]
    assert "at least 10 frames" in cases["tooShort"]
    assert "whole number" in cases["fractional"] and "whole number" in cases["negative"]


def test_video_decoder_flag_on_every_generate_task() -> None:
    result = _node(
        """const ctx = {frames: 49, fps: 24, width: 704, height: 448, seed: 1, inputMeta: () => ({})};
        const out = {};
        for (const task of t.LTX_TASKS.filter((x) => x.cmd === "generate")) {
          const fields = task.fields.map((f) => f.key);
          const on = task.build({videoDecoder: "diffusion", beats: [], anchors: []}, ctx);
          const conv = task.build({videoDecoder: "conv", beats: [], anchors: []}, ctx);
          const unset = task.build({beats: [], anchors: []}, ctx);
          out[task.id] = [fields.includes("videoDecoder"),
                          on.join(" ").includes("--video-decoder diffusion"),
                          conv.includes("--video-decoder"),
                          unset.includes("--video-decoder")];
        }
        console.log(JSON.stringify(out));"""
    )
    # The flag reaches every generate task, and conv -- explicit or unset -- emits nothing,
    # so an existing session's argv is unchanged.
    assert result and all(v == [True, True, False, False] for v in result.values()), result


def test_video_decoder_requirements() -> None:
    cases = _node(
        f"""const r = (v, model) => t.generateRequires(v, model, {{frames: 49}});
        const m25 = {json.dumps(MODEL_25)}, m23 = {json.dumps(MODEL_23)};
        const hf = {json.dumps(MODEL_HF)}, old25 = {json.dumps(MODEL_25_NO_AV)};
        console.log(JSON.stringify({{
          ok25: r({{videoDecoder: "diffusion"}}, m25),
          on23: r({{videoDecoder: "diffusion"}}, m23),
          conv23: r({{videoDecoder: "conv"}}, m23),
          unset23: r({{}}, m23),
          hf: r({{videoDecoder: "diffusion"}}, hf),
          old25: r({{videoDecoder: "diffusion"}}, old25),
        }}));"""
    )
    assert cases["ok25"] is None
    # conv is the default and must stay available everywhere.
    assert cases["conv23"] is None and cases["unset23"] is None
    # A repo id isn't inspected locally, so the CLI does the refusing.
    assert cases["hf"] is None
    assert "LTX-2.5" in cases["on23"]
    # A 2.5 pack without the weights is refused too -- is_25 alone is not the gate.
    assert "vae_decoder_av" in cases["old25"]


def test_video_decoder_refuses_live_preview() -> None:
    """A preview decodes once per step, which the diffusion decoder cannot do cheaply."""
    cases = _node(
        f"""const m25 = {json.dumps(MODEL_25)};
        const r = (v, ctx) => t.generateRequires(v, m25, ctx);
        console.log(JSON.stringify({{
          diffPreview: r({{videoDecoder: "diffusion"}}, {{frames: 49, preview: true}}),
          diffNoPreview: r({{videoDecoder: "diffusion"}}, {{frames: 49, preview: false}}),
          convPreview: r({{videoDecoder: "conv"}}, {{frames: 49, preview: true}}),
          unsetPreview: r({{}}, {{frames: 49, preview: true}}),
          noCtx: r({{videoDecoder: "diffusion"}}, {{}}),
        }}));"""
    )
    assert "preview" in cases["diffPreview"].lower()
    # conv keeps previews, and the default path is untouched.
    assert cases["diffNoPreview"] is None and cases["convPreview"] is None
    assert cases["unsetPreview"] is None and cases["noCtx"] is None


def test_video_decoder_select_renders_its_hint() -> None:
    """``renderField``'s select case must show ``hint``; it silently dropped it before."""
    app = (Path(__file__).resolve().parents[1] / "web" / "static" / "app.js").read_text()
    select_case = app.split('case "select": {', 1)[1].split('case "check":', 1)[0]
    assert "field-hint" in select_case
