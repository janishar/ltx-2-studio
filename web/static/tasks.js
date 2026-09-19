// ltx studio task catalog — one entry per way ltx-2-mlx can run.
//
// The server knows nothing about tasks: it receives a subcommand plus an
// argument list and appends --model / --gemma / --quantize-on-load / --output
// itself. Add or change a task by editing this file only.
//
// Task keys
//   id, category, label, description
//   cmd        ltx-2-mlx subcommand
//   output     "mp4" (a take) | "dir" (a folder in outputs/) | "none"
//   blocks     shared form blocks: prompt ("required"|"optional"), canvas,
//              duration ("auto" also offers the LTX-2.5 DurationHead), seed,
//              lowRam, tiling, quantize
//   fields     task-specific fields (types: select, number, text, textarea,
//              check, media, rows). `when: {key: [values]}` shows a field
//              conditionally; media fields pick from the session's inputs.
//   requires(v, model, ctx) -> reason string when the model can't run it, else null
//   build(v, ctx)           -> argument list (strings, or {input: name} tokens)
//
// ctx: {frames, fps, width, height, seed, autoDuration, inputMeta(name)}
//      (requires gets {frames, autoDuration} only)

const LORA_ROWS = {
  key: "loras", label: "LoRAs", type: "rows", addLabel: "Add LoRA",
  itemFields: [
    { key: "path", label: "Path or HF repo", type: "text", placeholder: "org/repo or /path/lora.safetensors" },
    { key: "strength", label: "Strength", type: "number", step: 0.05, default: 1.0 },
  ],
};

const GENERATE_ADVANCED = [
  { key: "pipeline", label: "Pipeline", type: "select", default: "distilled", advanced: false,
    options: [
      ["distilled", "Distilled — fastest (8 + 3 steps)"],
      ["two-stage", "Two-stage — dev + CFG (needs dev model)"],
      ["two-stages-hq", "Two-stage HQ — res_2s + CFG (needs dev model)"],
      ["one-stage", "One-stage — dev + CFG at full res (needs dev model)"],
    ] },
  { key: "generatedKeyframes", label: "Generated keyframes", type: "number", min: 0, max: 16, step: 1, placeholder: "0 = off",
    hint: "LTX-2.5 · extra keyframes at evenly spaced interior frames sharpen fast motion; each adds one latent frame of stage-1 tokens",
    advanced: false },
  { key: "videoDecoder", label: "Video decoder", type: "select", default: "conv", advanced: true,
    hint: "LTX-2.5 · diffusion is sharper on fine detail but several times slower, and capped at 512×768×49",
    options: [
      ["conv", "Conv — default"],
      ["diffusion", "Diffusion — sharper, slower"],
    ] },
  { key: "steps", label: "Steps", type: "number", min: 1, max: 100, placeholder: "8", when: { pipeline: ["one-stage"] }, advanced: true },
  { key: "stage1Steps", label: "Stage 1 steps", type: "number", min: 1, max: 100, placeholder: "30 / 15 HQ", when: { pipeline: ["two-stage", "two-stages-hq"] }, advanced: true },
  { key: "stage2Steps", label: "Stage 2 steps", type: "number", min: 1, max: 3, placeholder: "3", when: { pipeline: ["two-stage", "two-stages-hq"] }, advanced: true },
  { key: "cfg", label: "CFG scale", type: "number", step: 0.1, placeholder: "3.0", when: { pipeline: ["two-stage", "two-stages-hq", "one-stage"] }, advanced: true },
  { key: "stg", label: "STG scale", type: "number", step: 0.1, placeholder: "0.0", when: { pipeline: ["two-stage", "two-stages-hq", "one-stage"] }, advanced: true },
  { key: "devTransformer", label: "Dev transformer file", type: "text", placeholder: "transformer-dev.safetensors", when: { pipeline: ["two-stage", "two-stages-hq", "one-stage"] }, advanced: true },
  { key: "distilledLora", label: "Distilled LoRA", type: "text", placeholder: "pack default", when: { pipeline: ["two-stage", "two-stages-hq"] }, advanced: true },
  { key: "distilledLoraStrength", label: "Distilled LoRA strength", type: "number", step: 0.05, placeholder: "1.0", when: { pipeline: ["two-stage", "two-stages-hq"] }, advanced: true },
  { key: "teacache", label: "TeaCache stage-1 acceleration", type: "check", hint: "LTX-2.3 packs only", when: { pipeline: ["two-stage", "two-stages-hq"] }, advanced: true },
  { key: "teacacheThresh", label: "TeaCache threshold", type: "number", step: 0.1, placeholder: "0.5", when: { teacache: [true] }, advanced: true },
  { key: "enhancePrompt", label: "Enhance prompt with Gemma 3 first", type: "check", hint: "LTX-2.3 packs only", advanced: true },
  { key: "noAudio", label: "No audio track", type: "check", hint: "skip audio decode + mux", advanced: true },
  { ...LORA_ROWS, advanced: true },
];

const PIPELINE_FLAG = {
  "distilled": "--distilled", "two-stage": "--two-stage", "two-stages-hq": "--two-stages-hq", "one-stage": "--one-stage",
};

function num(v) { return v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v); }
function opt(args, flag, value) { if (num(value) !== null) args.push(flag, String(value)); }
function optText(args, flag, value) { if (value && String(value).trim()) args.push(flag, String(value).trim()); }
function flag(args, name, on) { if (on) args.push(name); }
function loraArgs(args, rows, name = "--lora") {
  for (const r of rows || []) if (r.path && r.path.trim()) args.push(name, r.path.trim(), String(num(r.strength) ?? 1.0));
}

function generateArgs(v, ctx) {
  const args = [PIPELINE_FLAG[v.pipeline || "distilled"]];
  if (v.pipeline === "one-stage") opt(args, "--steps", v.steps);
  if (v.pipeline === "two-stage" || v.pipeline === "two-stages-hq") {
    opt(args, "--stage1-steps", v.stage1Steps);
    opt(args, "--stage2-steps", v.stage2Steps);
    optText(args, "--distilled-lora", v.distilledLora);
    opt(args, "--distilled-lora-strength", v.distilledLoraStrength);
    flag(args, "--enable-teacache", v.teacache);
    if (v.teacache) opt(args, "--teacache-thresh", v.teacacheThresh);
  }
  if (v.pipeline && v.pipeline !== "distilled") {
    opt(args, "--cfg-scale", v.cfg);
    opt(args, "--stg-scale", v.stg);
    optText(args, "--dev-transformer", v.devTransformer);
  }
  if ((num(v.generatedKeyframes) || 0) > 0) args.push("--num-generated-keyframes", String(Math.round(num(v.generatedKeyframes))));
  if ((v.videoDecoder || "conv") !== "conv") args.push("--video-decoder", v.videoDecoder);
  flag(args, "--enhance-prompt", v.enhancePrompt);
  flag(args, "--no-audio", v.noAudio);
  loraArgs(args, v.loras);
  return args;
}

function generateRequires(v, model, ctx = {}) {
  if ((v.pipeline || "distilled") !== "distilled" && !model.has_dev) return "This pipeline needs the dev transformer, which the model doesn't have.";
  const keyframes = num(v.generatedKeyframes);
  if (keyframes !== null && keyframes !== 0) {
    if (!Number.isInteger(keyframes) || keyframes < 0) return "Generated keyframes must be a whole number (0 turns it off).";
    // A Hugging Face repo id isn't inspected (is_25 unknown); the CLI refuses it up front if unsupported.
    if (model.local && !model.is_25) return "Generated keyframes need an LTX-2.5 model (this is an LTX-2.3 pack).";
    if (!ctx.autoDuration && ctx.frames && ctx.frames < keyframes + 2) {
      return `${keyframes} generated keyframes need at least ${keyframes + 2} frames; this clip has ${ctx.frames}.`;
    }
  }
  if ((v.pipeline || "distilled") === "distilled" && !model.has_distilled) return "The distilled pipeline needs the distilled transformer.";
  if (v.enhancePrompt && model.is_25) return "--enhance-prompt uses Gemma 3 and is not supported on LTX-2.5 packs.";
  if (v.teacache && model.is_25) return "TeaCache is not calibrated for LTX-2.5.";
  // A Hugging Face repo id isn't inspected; the CLI refuses it up front if the pack lacks the weights.
  if ((v.videoDecoder || "conv") === "diffusion") {
    if (model.local && !model.has_diffusion_decoder) {
      return "The diffusion video decoder ships with LTX-2.5 models only (no vae_decoder_av weights here).";
    }
    // A preview decodes one window per step; the diffusion decoder takes tens of seconds to do that.
    if (ctx.preview) return "Live preview needs the conv decoder — turn preview off to use the diffusion decoder.";
  }
  return null;
}

const needsDev = (label) => (v, model) => (model.has_dev ? null : `${label} needs the dev transformer, which the model doesn't have.`);
const needs23 = (label) => (v, model) => (model.is_25 ? `${label} runs on LTX-2.3 packs only (no official LTX-2.5 IC-LoRAs yet).` : null);
const latent = (seconds, fps) => Math.max(0, Math.round((num(seconds) || 0) * fps / 8));

const LTX_TASKS = [
  // ── Generate ──────────────────────────────────────────────────────────
  {
    id: "t2v", category: "Generate", label: "Text → Video",
    description: "Generate video with synchronized audio from a prompt.",
    cmd: "generate", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: "auto", seed: true, lowRam: true, tiling: true, quantize: true },
    fields: GENERATE_ADVANCED,
    requires: generateRequires,
    build: (v, ctx) => generateArgs(v, ctx),
  },
  {
    id: "i2v", category: "Generate", label: "Image → Video",
    description: "Animate a start image. The first latent frame is replaced by the image.",
    cmd: "generate", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: "auto", seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { key: "image", label: "Start image", type: "media", accept: "image", required: true },
      { key: "strength", label: "Image strength", type: "number", step: 0.05, default: 1.0 },
      ...GENERATE_ADVANCED,
    ],
    requires: generateRequires,
    build: (v, ctx) => [...generateArgs(v, ctx), "--image", { input: v.image }, "0", String(num(v.strength) ?? 1.0)],
  },
  {
    id: "flf2v", category: "Generate", label: "First + Last Frame",
    description: "Anchor both ends of the clip and let the model animate the transition.",
    cmd: "generate", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { key: "first", label: "First frame", type: "media", accept: "image", required: true },
      { key: "last", label: "Last frame", type: "media", accept: "image", required: true },
      { key: "strength", label: "Anchor strength", type: "number", step: 0.05, default: 1.0 },
      ...GENERATE_ADVANCED,
    ],
    requires: generateRequires,
    build: (v, ctx) => [
      ...generateArgs(v, ctx),
      "--image", { input: v.first }, "0", String(num(v.strength) ?? 1.0),
      "--image", { input: v.last }, String(ctx.frames - 1), String(num(v.strength) ?? 1.0),
    ],
  },
  {
    id: "anchors", category: "Generate", label: "Multi-Image Anchors",
    description: "Place several images at chosen times. Time 0 replaces the first frame; later anchors guide softly.",
    cmd: "generate", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { key: "anchors", label: "Anchors", type: "rows", addLabel: "Add anchor", minRows: 1,
        itemFields: [
          { key: "image", label: "Image", type: "media", accept: "image", required: true },
          { key: "time", label: "At second", type: "number", step: 0.1, default: 0 },
          { key: "strength", label: "Strength", type: "number", step: 0.05, default: 1.0 },
        ] },
      ...GENERATE_ADVANCED,
    ],
    requires: generateRequires,
    build: (v, ctx) => {
      const args = generateArgs(v, ctx);
      for (const a of v.anchors || []) {
        if (!a.image) continue;
        const frame = Math.min(Math.round((num(a.time) || 0) * ctx.fps), ctx.frames - 1);
        args.push("--image", { input: a.image }, String(frame), String(num(a.strength) ?? 1.0));
      }
      return args;
    },
  },
  {
    id: "story", category: "Generate", label: "Prompt Beats",
    description: "Prompt Relay: sequence local prompts over time inside one generation. The main prompt applies throughout.",
    cmd: "generate", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, quantize: true },
    fields: [
      { key: "beats", label: "Beats (timeline order)", type: "rows", addLabel: "Add beat", minRows: 2,
        itemFields: [
          { key: "text", label: "What happens", type: "text", placeholder: "she stands up and walks to the window" },
          { key: "seconds", label: "Seconds (blank = auto)", type: "number", step: 0.5 },
        ] },
      { key: "startImage", label: "Optional start image", type: "media", accept: "image" },
      { key: "relayEpsilon", label: "Relay epsilon", type: "number", step: 0.0001, placeholder: "0.001", advanced: true },
      { key: "relayStrength", label: "Relay strength", type: "number", step: 0.1, placeholder: "1.0", advanced: true },
      ...GENERATE_ADVANCED,
    ],
    requires: generateRequires,
    build: (v, ctx) => {
      const args = generateArgs(v, ctx);
      for (const b of v.beats || []) {
        if (!b.text || !b.text.trim()) continue;
        args.push("--segment", b.text.trim());
        if (num(b.seconds) !== null) args.push(String(Math.max(1, latent(b.seconds, ctx.fps))));
      }
      opt(args, "--relay-epsilon", v.relayEpsilon);
      opt(args, "--relay-strength", v.relayStrength);
      if (v.startImage) args.push("--image", { input: v.startImage }, "0", "1.0");
      return args;
    },
  },

  // ── Audio ─────────────────────────────────────────────────────────────
  {
    id: "a2v", category: "Audio", label: "Audio → Video",
    description: "Generate video driven by an audio file (music, speech, sound effects). Dev model + CFG.",
    cmd: "a2v", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { key: "audio", label: "Audio", type: "media", accept: "audio", required: true },
      { key: "audioStart", label: "Start at second", type: "number", step: 0.1, default: 0 },
      { key: "image", label: "Optional start image", type: "media", accept: "image" },
      { key: "stage1Steps", label: "Stage 1 steps", type: "number", placeholder: "30", advanced: true },
      { key: "stage2Steps", label: "Stage 2 steps", type: "number", placeholder: "3", advanced: true },
      { key: "cfg", label: "CFG scale", type: "number", step: 0.1, placeholder: "3.0", advanced: true },
      { key: "stg", label: "STG scale", type: "number", step: 0.1, placeholder: "0.0", advanced: true },
    ],
    requires: needsDev("Audio → Video"),
    build: (v) => {
      const args = ["--audio", { input: v.audio }];
      if (num(v.audioStart)) args.push("--audio-start", String(v.audioStart));
      if (v.image) args.push("--image", { input: v.image }, "0", "1.0");
      opt(args, "--stage1-steps", v.stage1Steps);
      opt(args, "--stage2-steps", v.stage2Steps);
      opt(args, "--cfg-scale", v.cfg);
      opt(args, "--stg-scale", v.stg);
      return args;
    },
  },

  // ── Edit video ────────────────────────────────────────────────────────
  {
    id: "retake", category: "Edit Video", label: "Retake a Section",
    description: "Regenerate a time range of an existing video, keeping the rest. Dev model + CFG.",
    cmd: "retake", output: "mp4",
    blocks: { prompt: "required", seed: true, lowRam: true, quantize: true },
    fields: [
      { key: "video", label: "Source video", type: "media", accept: "video", required: true },
      { key: "from", label: "From second", type: "number", step: 0.1, default: 0 },
      { key: "to", label: "To second (blank = end)", type: "number", step: 0.1 },
      { key: "keepAudio", label: "Keep original audio", type: "check" },
      { key: "steps", label: "Steps", type: "number", placeholder: "30", advanced: true },
      { key: "cfg", label: "CFG scale", type: "number", step: 0.1, placeholder: "3.0", advanced: true },
      { key: "stg", label: "STG scale", type: "number", step: 0.1, placeholder: "0.0", advanced: true },
    ],
    requires: needsDev("Retake"),
    build: (v, ctx) => {
      const meta = ctx.inputMeta(v.video);
      const fps = meta.fps || ctx.fps;
      const start = latent(v.from, fps);
      const toSeconds = num(v.to) ?? meta.duration ?? (num(v.from) || 0) + 1;
      const end = Math.max(latent(toSeconds, fps), start + 1);
      const args = ["--video", { input: v.video }, "--start", String(start), "--end", String(end)];
      flag(args, "--no-regen-audio", v.keepAudio);
      opt(args, "--steps", v.steps);
      opt(args, "--cfg-scale", v.cfg);
      opt(args, "--stg-scale", v.stg);
      return args;
    },
  },
  {
    id: "extend", category: "Edit Video", label: "Extend a Video",
    description: "Add seconds before or after an existing video. Dev model + CFG.",
    cmd: "extend", output: "mp4",
    blocks: { prompt: "required", seed: true, lowRam: true, quantize: true },
    fields: [
      { key: "video", label: "Source video", type: "media", accept: "video", required: true },
      { key: "addSeconds", label: "Seconds to add", type: "number", step: 0.5, default: 2 },
      { key: "direction", label: "Direction", type: "select", default: "after", options: [["after", "After (continue)"], ["before", "Before (prequel)"]] },
      { key: "steps", label: "Steps", type: "number", placeholder: "30", advanced: true },
      { key: "cfg", label: "CFG scale", type: "number", step: 0.1, placeholder: "3.0", advanced: true },
      { key: "stg", label: "STG scale", type: "number", step: 0.1, placeholder: "0.0", advanced: true },
    ],
    requires: needsDev("Extend"),
    build: (v, ctx) => {
      const fps = ctx.inputMeta(v.video).fps || ctx.fps;
      const frames = Math.max(1, Math.ceil((num(v.addSeconds) || 1) * fps / 8));
      const args = ["--video", { input: v.video }, "--extend-frames", String(frames), "--direction", v.direction || "after"];
      opt(args, "--steps", v.steps);
      opt(args, "--cfg-scale", v.cfg);
      opt(args, "--stg-scale", v.stg);
      return args;
    },
  },
  {
    id: "keyframe", category: "Edit Video", label: "Keyframe Interpolation",
    description: "Interpolate between a start and end image with the dev model + CFG (more faithful than First + Last Frame).",
    cmd: "keyframe", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { key: "start", label: "Start keyframe", type: "media", accept: "image", required: true },
      { key: "end", label: "End keyframe", type: "media", accept: "image", required: true },
      { key: "startStrength", label: "Start strength", type: "number", step: 0.05, placeholder: "1.0" },
      { key: "endStrength", label: "End strength", type: "number", step: 0.05, placeholder: "1.0" },
      { key: "stage1Steps", label: "Stage 1 steps", type: "number", placeholder: "30", advanced: true },
      { key: "stage2Steps", label: "Stage 2 steps", type: "number", placeholder: "3", advanced: true },
      { key: "cfg", label: "CFG scale", type: "number", step: 0.1, placeholder: "3.0", advanced: true },
      { key: "stg", label: "STG scale", type: "number", step: 0.1, placeholder: "0.0", advanced: true },
      { key: "devTransformer", label: "Dev transformer file", type: "text", placeholder: "transformer-dev.safetensors", advanced: true },
      { key: "distilledLora", label: "Distilled LoRA", type: "text", placeholder: "pack default", advanced: true },
      { key: "loraStrength", label: "Distilled LoRA strength", type: "number", step: 0.05, placeholder: "1.0", advanced: true },
    ],
    requires: needsDev("Keyframe interpolation"),
    build: (v) => {
      const args = ["--start", { input: v.start }, "--end", { input: v.end }];
      opt(args, "--start-strength", v.startStrength);
      opt(args, "--end-strength", v.endStrength);
      opt(args, "--stage1-steps", v.stage1Steps);
      opt(args, "--stage2-steps", v.stage2Steps);
      opt(args, "--cfg-scale", v.cfg);
      opt(args, "--stg-scale", v.stg);
      optText(args, "--dev-transformer", v.devTransformer);
      optText(args, "--distilled-lora", v.distilledLora);
      opt(args, "--lora-strength", v.loraStrength);
      return args;
    },
  },

  // ── Control (IC-LoRA) ─────────────────────────────────────────────────
  {
    id: "ic-lora", category: "Control (IC-LoRA)", label: "Control Video → Video",
    description: "Follow a control video (depth, canny, pose, motion tracks) with an IC-LoRA such as Union Control.",
    cmd: "ic-lora", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { ...LORA_ROWS, minRows: 1, defaultRows: [{ path: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", strength: 1.0 }] },
      { key: "controls", label: "Control videos", type: "rows", addLabel: "Add control", minRows: 1,
        itemFields: [
          { key: "video", label: "Control video", type: "media", accept: "video", required: true },
          { key: "strength", label: "Strength", type: "number", step: 0.05, default: 1.0 },
        ] },
      { key: "image", label: "Optional start image", type: "media", accept: "image" },
      { key: "topology", label: "Topology", type: "select", default: "two-stage",
        options: [["two-stage", "Two-stage (default)"], ["skip-stage-2", "Skip stage 2 (half-res)"], ["upsample-only", "Upsample only (fast draft)"], ["single-stage", "Single stage full-res"]] },
      { key: "refineSteps", label: "Control-aware refine steps", type: "number", min: 1, max: 8, when: { topology: ["upsample-only"] } },
      { key: "conditioningStrength", label: "Conditioning attention strength", type: "number", step: 0.05, placeholder: "1.0", advanced: true },
      { key: "stage1Steps", label: "Stage 1 steps", type: "number", placeholder: "8", advanced: true },
      { key: "stage2Steps", label: "Stage 2 steps", type: "number", placeholder: "3", advanced: true },
      { key: "devTransformer", label: "Dev transformer file (dev mode)", type: "text", placeholder: "blank = distilled", advanced: true },
      { key: "distilledLora", label: "Distilled LoRA (dev mode)", type: "text", advanced: true },
      { key: "distilledLoraStrength", label: "Distilled LoRA strength", type: "number", step: 0.05, placeholder: "0.5", advanced: true },
    ],
    requires: needs23("IC-LoRA"),
    build: (v) => {
      const args = [];
      loraArgs(args, v.loras);
      for (const c of v.controls || []) if (c.video) args.push("--video-conditioning", { input: c.video }, String(num(c.strength) ?? 1.0));
      if (v.image) args.push("--image", { input: v.image }, "0", "1.0");
      if (v.topology && v.topology !== "two-stage") args.push(`--${v.topology}`);
      if (v.topology === "upsample-only") opt(args, "--refine-steps", v.refineSteps);
      opt(args, "--conditioning-strength", v.conditioningStrength);
      opt(args, "--stage1-steps", v.stage1Steps);
      opt(args, "--stage2-steps", v.stage2Steps);
      optText(args, "--dev-transformer", v.devTransformer);
      optText(args, "--distilled-lora", v.distilledLora);
      opt(args, "--distilled-lora-strength", v.distilledLoraStrength);
      return args;
    },
  },
  {
    id: "hdr-ic-lora", category: "Control (IC-LoRA)", label: "HDR Video",
    description: "Linear HDR output via the HDR IC-LoRA — upgrade an SDR video, or text-to-HDR without a source. Also writes a .hdr.npz.",
    cmd: "hdr-ic-lora", output: "mp4",
    blocks: { prompt: "required", canvas: true, duration: true, seed: true, lowRam: true, tiling: true, quantize: true },
    fields: [
      { ...LORA_ROWS, minRows: 1, defaultRows: [{ path: "Lightricks/LTX-2.3-22b-IC-LoRA-HDR", strength: 1.0 }] },
      { key: "source", label: "Optional SDR source video", type: "media", accept: "video" },
      { key: "sourceStrength", label: "Source strength", type: "number", step: 0.05, default: 1.0 },
      { key: "image", label: "Optional start image", type: "media", accept: "image" },
      { key: "skipStage2", label: "Skip stage 2 (half-res)", type: "check" },
      { key: "conditioningStrength", label: "Conditioning attention strength", type: "number", step: 0.05, placeholder: "1.0", advanced: true },
      { key: "stage1Steps", label: "Stage 1 steps", type: "number", placeholder: "8", advanced: true },
      { key: "stage2Steps", label: "Stage 2 steps", type: "number", placeholder: "3", advanced: true },
    ],
    requires: needs23("HDR IC-LoRA"),
    build: (v) => {
      const args = [];
      loraArgs(args, v.loras);
      if (v.source) args.push("--video-conditioning", { input: v.source }, String(num(v.sourceStrength) ?? 1.0));
      if (v.image) args.push("--image", { input: v.image }, "0", "1.0");
      flag(args, "--skip-stage-2", v.skipStage2);
      opt(args, "--conditioning-strength", v.conditioningStrength);
      opt(args, "--stage1-steps", v.stage1Steps);
      opt(args, "--stage2-steps", v.stage2Steps);
      return args;
    },
  },
  {
    id: "lipdub", category: "Control (IC-LoRA)", label: "Lip Dub",
    description: "Re-sync a reference video's lips to its audio with the LipDub IC-LoRA. Output audio is a VAE reconstruction.",
    cmd: "lipdub", output: "mp4",
    blocks: { prompt: "required", canvas: true, seed: true, lowRam: true, quantize: true },
    fields: [
      { key: "reference", label: "Reference video (with audio)", type: "media", accept: "video", required: true },
      { key: "referenceStrength", label: "Reference strength", type: "number", step: 0.05, placeholder: "1.0" },
      { key: "lora", label: "LipDub IC-LoRA", type: "text", default: "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub" },
      { key: "loraStrength", label: "LoRA strength", type: "number", step: 0.05, default: 1.0 },
      { key: "stage1Steps", label: "Stage 1 steps", type: "number", placeholder: "8", advanced: true },
      { key: "stage2Steps", label: "Stage 2 steps", type: "number", placeholder: "3", advanced: true },
    ],
    requires: needs23("Lip dub"),
    build: (v) => {
      const args = ["--reference-video", { input: v.reference }];
      opt(args, "--reference-strength", v.referenceStrength);
      if (v.lora) args.push("--lora", v.lora, String(num(v.loraStrength) ?? 1.0));
      opt(args, "--stage1-steps", v.stage1Steps);
      opt(args, "--stage2-steps", v.stage2Steps);
      return args;
    },
  },

  // ── Tools ─────────────────────────────────────────────────────────────
  {
    id: "enhance", category: "Tools", label: "Enhance Prompt",
    description: "Rewrite a prompt with Gemma 3 (downloads mlx-community/gemma-3-12b-it-4bit on first use). The result appears in the terminal.",
    cmd: "enhance", output: "none",
    blocks: { prompt: "required", seed: true },
    fields: [
      { key: "mode", label: "Prompt mode", type: "select", default: "t2v", options: [["t2v", "Text → video"], ["i2v", "Image → video"]] },
    ],
    requires: () => null,
    build: (v) => ["--mode", v.mode || "t2v"],
  },
  {
    id: "info", category: "Tools", label: "Model Info",
    description: "List the model's weight files, sizes and configuration.",
    cmd: "info", output: "none",
    blocks: {},
    fields: [],
    requires: () => null,
    build: () => [],
  },

  // ── Training ──────────────────────────────────────────────────────────
  {
    id: "slice", category: "Training", label: "Slice Clips",
    description: "Cut source videos into fixed-length training clips (writes a folder into this session's outputs).",
    cmd: "slice", output: "dir",
    blocks: {},
    fields: [
      { key: "sources", label: "Source video files or directories (one per line, server paths)", type: "textarea", rows: 3, required: true },
      { key: "interval", label: "Clip length (s)", type: "number", step: 0.5, placeholder: "4.0" },
      { key: "res", label: "Resolution WxH", type: "text", placeholder: "384x384" },
      { key: "fps", label: "FPS", type: "number", placeholder: "24" },
      { key: "fit", label: "Aspect", type: "select", default: "crop", options: [["crop", "Center crop"], ["pad", "Letterbox"]] },
      { key: "minLength", label: "Min length (s)", type: "number", step: 0.5, placeholder: "2.0", advanced: true },
      { key: "maxClips", label: "Max clips per source", type: "number", advanced: true },
      { key: "sample", label: "Sampling (with max clips)", type: "select", default: "even", options: [["even", "Spread evenly"], ["sequential", "First N"]], advanced: true },
      { key: "skipStart", label: "Skip start (s)", type: "number", step: 0.5, advanced: true },
      { key: "skipEnd", label: "Skip end (s)", type: "number", step: 0.5, advanced: true },
      { key: "captionTemplate", label: "Caption template", type: "text", advanced: true },
      { key: "timecodes", label: "Timecodes file (server path)", type: "text", advanced: true },
      { key: "crf", label: "x264 CRF", type: "number", placeholder: "18", advanced: true },
    ],
    requires: () => null,
    build: (v) => {
      const args = [];
      opt(args, "--interval", v.interval);
      optText(args, "--res", v.res);
      opt(args, "--fps", v.fps);
      optText(args, "--fit", v.fit);
      opt(args, "--min-length", v.minLength);
      opt(args, "--max-clips", v.maxClips);
      if (num(v.maxClips) !== null) optText(args, "--sample", v.sample);
      opt(args, "--skip-start", v.skipStart);
      opt(args, "--skip-end", v.skipEnd);
      optText(args, "--caption-template", v.captionTemplate);
      optText(args, "--timecodes", v.timecodes);
      opt(args, "--crf", v.crf);
      for (const line of (v.sources || "").split("\n")) if (line.trim()) args.push({ path: line.trim() });
      return args;
    },
  },
  {
    id: "preprocess", category: "Training", label: "Preprocess Dataset",
    description: "Encode a folder of clips + captions into latents for training (writes a folder into this session's outputs).",
    cmd: "preprocess", output: "dir",
    blocks: {},
    fields: [
      { key: "videos", label: "Videos directory (server path)", type: "text", required: true },
      { key: "captions", label: "Captions directory (server path)", type: "text" },
      { key: "captionExt", label: "Caption extension", type: "text", placeholder: ".txt", advanced: true },
      { key: "width", label: "Resize width", type: "number", step: 32 },
      { key: "height", label: "Resize height", type: "number", step: 32 },
      { key: "maxFrames", label: "Max frames (8k+1)", type: "number", placeholder: "97" },
      { key: "withAudio", label: "Also encode audio latents", type: "check" },
      { key: "frameRate", label: "Override frame rate", type: "number", advanced: true },
    ],
    requires: () => null,
    build: (v) => {
      const args = ["--videos", { path: v.videos }];
      if (v.captions) args.push("--captions", { path: v.captions });
      optText(args, "--caption-ext", v.captionExt);
      opt(args, "--width", v.width);
      opt(args, "--height", v.height);
      opt(args, "--max-frames", v.maxFrames);
      flag(args, "--with-audio", v.withAudio);
      opt(args, "--frame-rate", v.frameRate);
      return args;
    },
  },
  {
    id: "train", category: "Training", label: "Train LoRA",
    description: "Run training from a YAML config (see packages/ltx-trainer/configs/). Output paths come from the config.",
    cmd: "train", output: "none",
    blocks: {},
    fields: [
      { key: "config", label: "Config YAML (server path)", type: "text", required: true, placeholder: "packages/ltx-trainer/configs/lora_t2v.yaml" },
      { key: "lowRamTrain", label: "Gradient checkpointing (--low-ram)", type: "check" },
    ],
    requires: () => null,
    build: (v) => {
      const args = ["--config", { path: v.config }];
      flag(args, "--low-ram", v.lowRamTrain);
      return args;
    },
  },
];

const SIZE_PRESETS = [
  ["small", "Small", 704, 448],
  ["square", "Square", 768, 768],
  ["portrait", "Portrait", 448, 704],
  ["sd", "SD", 896, 512],
  ["720p", "720p", 1280, 704],
  ["1080p", "1080p", 1920, 1088],
];

if (typeof module !== "undefined") {
  module.exports = { LTX_TASKS, generateArgs, generateRequires, num };
}
