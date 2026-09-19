// ltx studio front end — vanilla JS, no build step.
// Task definitions live in tasks.js; this file renders them, builds the
// ltx-2-mlx argument list, talks to server.py, and puts helmstudio's
// components on the page.

"use strict";

const TASKS = Object.fromEntries(LTX_TASKS.map((t) => [t.id, t]));
// Subcommands whose pipelines accept the --stepwise-* live preview flags.
const PREVIEW_COMMANDS = new Set(["generate", "a2v", "retake", "extend", "keyframe", "ic-lora", "hdr-ic-lora", "lipdub"]);

const S = {
  model: {},
  session: null,
  sessions: [],
  preferences: {},       // side panel, terminal height, notifications: helmstudio keeps them (/api/config)
  inputs: [],
  takes: [],
  taskId: "t2v",
  values: {},
  common: {
    prompt: "", width: 704, height: 448, frames: 49, fps: 24,
    // Canvas sizing: a named ratio ("16:9"), "input" (match the task's input
    // media) or "custom" (customRatio); megapixels is the target size.
    aspect: "custom", customRatio: 704 / 448, megapixels: 0.32,
    autoDuration: false, autoMin: 1, autoMax: 8, seed: 42,
    quantize: "8", lowRam: false, tileFrames: 1, tileSpatial: 1, tileOverlap: 2,
    extraArgs: "", takeName: "",
    preview: { enabled: false, interval: 1, frames: 8, position: "middle", frame: 0 },
  },
  queue: [],
  selectedTake: null,
  timeline: [],
  selectedTimeline: null,
  selectedSequence: null,  // the helmstudio sequence the viewer is playing (takes.js)
  followPreview: true,
  livePreview: null,
  scrub: null,
  restoring: false,
  runningId: null,
  estimate: null,        // pre-render estimate for the current form, from /api/estimate
  compare: [],           // take names picked for the seed grid / A/B wipe
  compareBatch: null,    // {ids: Set, outputs: []} while "Queue 3 seeds" runs
  batchOpening: false,   // the seed grid is about to open; don't auto-select a take
  inputFilter: "all",
};

// num() comes from tasks.js; $, el, api, debounce, fmtSecs, toast, … from util.js.

function inputByName(name) { return S.inputs.find((i) => i.name === name); }
function inputMeta(name) { const i = inputByName(name); return (i && i.probe) || {}; }

// ── task values ──────────────────────────────────────────────────────────

function defaultsFor(task) {
  const v = {};
  for (const f of task.fields) {
    if (f.type === "rows") {
      v[f.key] = (f.defaultRows || []).map((r) => ({ ...r }));
      while (v[f.key].length < (f.minRows || 0)) v[f.key].push(rowDefaults(f));
    } else if (f.default !== undefined) v[f.key] = f.default;
  }
  return v;
}

function rowDefaults(field) {
  const row = {};
  for (const f of field.itemFields) if (f.default !== undefined) row[f.key] = f.default;
  return row;
}

function taskValues(taskId = S.taskId) {
  if (!S.values[taskId]) S.values[taskId] = defaultsFor(TASKS[taskId]);
  return S.values[taskId];
}

function visible(field, v) {
  if (!field.when) return true;
  return Object.entries(field.when).every(([k, vals]) => {
    const def = TASKS[S.taskId].fields.find((f) => f.key === k);
    const current = v[k] ?? (def ? def.default : undefined) ?? false;
    return vals.includes(current);
  });
}

// ── rendering: task form ─────────────────────────────────────────────────

function renderTaskSelect() {
  const select = $("taskSelect");
  select.innerHTML = "";
  const groups = {};
  for (const t of LTX_TASKS) (groups[t.category] ||= []).push(t);
  for (const [category, tasks] of Object.entries(groups)) {
    const group = el("optgroup", { label: category });
    for (const t of tasks) group.append(el("option", { value: t.id, text: t.label }));
    select.append(group);
  }
  select.value = S.taskId;
}

function renderTask() {
  const task = TASKS[S.taskId];
  const v = taskValues();
  const blocks = task.blocks || {};
  $("taskSelect").value = task.id;
  $("taskCmd").textContent = `ltx-2-mlx ${task.cmd}`;
  $("taskDescription").textContent = task.description;
  $("promptBlock").hidden = !blocks.prompt;
  $("canvasBlock").hidden = !blocks.canvas;
  $("durationBlock").hidden = !blocks.duration;
  $("autoDurationRow").hidden = blocks.duration !== "auto";
  $("seedBlock").hidden = !blocks.seed;
  $("quantizeField").hidden = !blocks.quantize;
  $("lowRamField").hidden = !blocks.lowRam;
  $("tilingFields").hidden = !blocks.tiling;
  $("performanceFields").hidden = !(blocks.quantize || blocks.lowRam || blocks.tiling);

  const main = task.fields.filter((f) => !f.advanced && visible(f, v));
  const advanced = task.fields.filter((f) => f.advanced && visible(f, v));
  $("taskFieldsBlock").hidden = main.length === 0;
  $("taskFields").replaceChildren(el("div", { class: "taskgrid" }, main.map((f) => renderField(f, v, () => renderTask()))));
  $("advancedFields").replaceChildren(
    advanced.length ? el("div", { class: "taskgrid" }, advanced.map((f) => renderField(f, v, () => renderTask()))) : "",
  );
  renderAvailability();
  fitCanvas();
  renderCanvas();
  renderPreviewOptions();
  refreshPreview();
}

function renderPreviewOptions() {
  const p = S.common.preview;
  const supported = PREVIEW_COMMANDS.has(TASKS[S.taskId].cmd);
  $("previewOptions").hidden = !supported;
  $("previewEnabled").checked = p.enabled;
  $("previewInterval").value = p.interval;
  $("previewFrames").value = String(p.frames);
  $("previewPosition").value = p.position;
  $("previewFrame").value = p.frame;
  $("previewFrameField").hidden = p.position !== "custom";
  $("previewSettings").classList.toggle("disabled", !p.enabled);
  const pixelFrames = 8 * p.frames - 7;
  const seconds = pixelFrames / (S.common.fps || 24);
  const steps = p.interval === 1 ? "every step" : `every ${p.interval} steps (and the last)`;
  $("previewCost").textContent = p.frames === 1
    ? `A still frame ${steps}. Cheapest option; no motion.`
    : `${pixelFrames} frames (${seconds.toFixed(1)} s) ${steps}. Keeps the VAE decoder in memory during denoising; with --low-ram it undoes most of the saving.`;
}

function previewRequest() {
  const p = S.common.preview;
  if (!p.enabled || !PREVIEW_COMMANDS.has(TASKS[S.taskId].cmd)) return undefined;
  const frame = { middle: null, start: 0, end: -1, custom: p.frame }[p.position];
  return { interval: p.interval, frames: p.frames, frame };
}

function renderField(field, values, rerender) {
  const label = el("span", {}, field.label, field.required ? el("span", { class: "req", text: " *" }) : null);
  const set = (value, structural) => {
    values[field.key] = value;
    if (structural) rerender();
    else refreshPreview();
    saveSettings();
  };
  switch (field.type) {
    case "select": {
      const select = el("select", { onchange: (e) => set(e.target.value, true) },
        field.options.map(([value, text]) => el("option", { value, text })));
      select.value = values[field.key] ?? field.default ?? field.options[0][0];
      return el("label", { class: `field${field.hint ? " full" : ""}` }, label, select,
        field.hint ? el("em", { class: "field-hint", text: field.hint }) : null);
    }
    case "check":
      return el("label", { class: "check full" },
        el("input", { type: "checkbox", checked: !!values[field.key], onchange: (e) => set(e.target.checked, true) }),
        field.label, field.hint ? el("em", { text: field.hint }) : null);
    case "number":
      return el("label", { class: `field${field.hint ? " full" : ""}` }, label,
        el("input", { type: "number", value: values[field.key] ?? "", step: field.step ?? "any", min: field.min, max: field.max,
          placeholder: field.placeholder ?? "", oninput: (e) => set(e.target.value === "" ? "" : Number(e.target.value)) }),
        field.hint ? el("em", { class: "field-hint", text: field.hint }) : null);
    case "text":
      return el("label", { class: "field" }, label,
        el("input", { type: "text", value: values[field.key] ?? "", placeholder: field.placeholder ?? "", spellcheck: false,
          oninput: (e) => set(e.target.value) }));
    case "textarea":
      return el("label", { class: "field" }, label,
        el("textarea", { rows: field.rows || 3, placeholder: field.placeholder ?? "", spellcheck: false,
          value: values[field.key] ?? "", oninput: (e) => set(e.target.value) }));
    case "media":
      return renderMedia(field, values, set, label);
    case "rows":
      return renderRows(field, values, rerender);
    default:
      return el("span", { text: `unknown field ${field.key}` });
  }
}

function renderMedia(field, values, set, label) {
  const current = values[field.key] || "";
  const compatible = S.inputs.filter((i) => field.accept === "file" || i.kind === field.accept);
  const item = inputByName(current);
  let preview = el("span", { text: field.accept });
  if (item && item.kind === "image") preview = el("img", { src: item.url, alt: "" });
  else if (item && item.kind === "video") preview = el("img", { src: item.thumb, alt: "" });
  else if (item && item.kind === "audio") preview = el("span", { text: "♪ audio" });
  const select = el("select", { onchange: (e) => set(e.target.value, true) },
    el("option", { value: "", text: compatible.length ? `Choose ${field.accept}…` : `Upload a ${field.accept} above` }),
    compatible.map((i) => el("option", { value: i.name, text: describeInput(i) })));
  select.value = current;
  const missing = current && !item;
  return el("div", { class: "media" }, label,
    el("div", { class: `slot${item ? " set" : ""}${missing ? " missing" : ""}` }, el("div", { class: "preview" }, preview), select));
}

function describeInput(i) {
  const p = i.probe || {};
  const bits = [i.name];
  if (p.width) bits.push(`${p.width}×${p.height}`);
  if (p.duration && i.kind !== "image") bits.push(`${p.duration.toFixed(1)}s`);
  return bits.join(" · ");
}

function renderRows(field, values, rerender) {
  const rows = (values[field.key] ||= []);
  const wrap = el("div", { class: "rows" },
    el("div", { class: "rowhead" }, el("span", { text: field.label }),
      el("button", { class: "ghost sm", type: "button", text: field.addLabel || "Add",
        onclick: () => { rows.push(rowDefaults(field)); rerender(); saveSettings(); } })));
  rows.forEach((row, index) => {
    const children = field.itemFields.map((f) => renderField(f, row, rerender));
    const canRemove = rows.length > (field.minRows || 0);
    wrap.append(el("div", { class: "row" }, children,
      canRemove ? el("button", { class: "remove", type: "button", title: "Remove", text: "×",
        onclick: () => { rows.splice(index, 1); rerender(); saveSettings(); } }) : null));
  });
  return wrap;
}

/** What task.requires() may check beyond the task values: the clip length (unless auto duration decides it) and whether live preview is on. */
function requiresContext(task) {
  const auto = (task.blocks || {}).duration === "auto" && S.common.autoDuration;
  const preview = PREVIEW_COMMANDS.has(task.cmd) && S.common.preview.enabled;
  return { frames: S.common.frames, autoDuration: auto, preview };
}

function renderAvailability() {
  const task = TASKS[S.taskId];
  const reason = S.model.configured ? task.requires(taskValues(), S.model, requiresContext(task)) : "No model configured — click Model in the top bar.";
  $("taskUnavailable").hidden = !reason;
  $("taskUnavailable").textContent = reason || "";
}

// ── canvas sizing (helpers in canvas.js) ─────────────────────────────────

function currentGrid() {
  return canvasGrid(TASKS[S.taskId].cmd, taskValues().pipeline || "distilled");
}

/** Size of the first image/video the task has selected, or null. */
function inputAspect() {
  const task = TASKS[S.taskId];
  const v = taskValues();
  const sized = (name) => {
    const p = name ? inputMeta(name) : {};
    return p.width && p.height ? { ratio: p.width / p.height, width: p.width, height: p.height } : null;
  };
  for (const f of task.fields) {
    if (f.type === "media" && f.accept !== "audio" && sized(v[f.key])) return sized(v[f.key]);
    if (f.type === "rows") {
      const media = f.itemFields.filter((i) => i.type === "media" && i.accept !== "audio");
      for (const row of v[f.key] || []) for (const i of media) if (sized(row[i.key])) return sized(row[i.key]);
    }
  }
  return null;
}

function canvasRatio() {
  const c = S.common;
  if (c.aspect === "input") {
    const a = inputAspect();
    if (a) return a.ratio;
  }
  return aspectRatioFor(c.aspect) || c.customRatio || c.width / c.height;
}

function syncCanvasInputs() {
  $("width").value = S.common.width;
  $("height").value = S.common.height;
}

/** Re-solve width/height from the ratio and megapixel target on the current grid. */
function solveCurrentCanvas() {
  const c = S.common;
  const solved = solveCanvas(canvasRatio(), c.megapixels, currentGrid());
  if (!solved || (solved.width === c.width && solved.height === c.height)) return false;
  c.width = solved.width;
  c.height = solved.height;
  syncCanvasInputs();
  return true;
}

/** Keep the canvas legal for the current task: re-solve ratio-driven sizes, snap custom ones. */
function fitCanvas() {
  const task = TASKS[S.taskId];
  if (!(task.blocks && task.blocks.canvas)) return;
  const c = S.common;
  const grid = currentGrid();
  let changed = false;
  // "Match input" with no sized input yet (nothing picked, or inputs still loading) keeps the size.
  const ratioDriven = c.aspect !== "custom" && (c.aspect !== "input" || inputAspect());
  if (ratioDriven) changed = solveCurrentCanvas();
  else if (c.width % grid || c.height % grid) {
    c.width = snapDimension(c.width, grid);
    c.height = snapDimension(c.height, grid);
    syncCanvasInputs();
    changed = true;
  }
  if (changed) saveSettings();
}

/** Take explicit dimensions (preset, typed size, restored take) and derive ratio + megapixels. */
function setCanvasDims(width, height) {
  const c = S.common;
  c.width = width;
  c.height = height;
  c.aspect = matchAspect(width, height, 0.005) || "custom";
  c.customRatio = width / height;
  c.megapixels = clampMegapixels((width * height) / 1e6);
  syncCanvasInputs();
}

function renderCanvas() {
  const task = TASKS[S.taskId];
  const c = S.common;
  const { width, height } = c;
  const grid = currentGrid();
  const input = inputAspect();

  const customRatio = c.aspect === "custom" && c.customRatio ? c.customRatio : width / height;
  const options = [el("option", { value: "custom", text: `Custom · ${customRatio.toFixed(3)}` })];
  if (input || c.aspect === "input") {
    options.push(el("option", { value: "input", text: input ? `Match input · ${input.width}×${input.height}` : "Match input (none selected)" }));
  }
  for (const [key, label] of ASPECT_RATIOS) options.push(el("option", { value: key, text: label }));
  $("aspectSelect").replaceChildren(...options);
  $("aspectSelect").value = c.aspect;

  $("megapixels").value = String(c.megapixels);
  $("megapixelsValue").textContent = `${Number(c.megapixels).toFixed(2)} MP`;
  $("resolvedDimensions").textContent = `${width} × ${height}`;
  $("actualMegapixels").textContent = `${((width * height) / 1e6).toFixed(2)} MP`;
  $("actualRatio").textContent = height ? (width / height).toFixed(3) : "–";
  $("latentSize").textContent = `${Math.floor(width / 32)} × ${Math.floor(height / 32)}`;
  $("canvasGridLabel").textContent = `×${grid} grid`;
  $("canvasGridLabel").title = grid === 64
    ? "Two-stage pipelines render half size first, so sizes step in multiples of 64"
    : "One-stage renders at full size, so sizes step in multiples of 32";
  $("width").step = $("height").step = $("width").min = $("height").min = String(grid);

  let message = "";
  if (task.blocks && task.blocks.canvas) {
    if (width % grid || height % grid) message = `Width and height snap to multiples of ${grid} for this pipeline.`;
    else if (width * height > CANVAS_MAX_PIXELS) message = "Beyond 1080p memory grows fast — consider tiling in Advanced.";
    else if (width * height > 1280 * 704) message = "Above 720p, memory grows quickly with duration — keep clips short or enable tiling.";
  }
  $("sizeWarn").hidden = !message;
  $("sizeWarn").textContent = message;

  // The pipelines resize and centre-crop conditioning media to the canvas.
  let aspectMessage = "";
  if (task.blocks && task.blocks.canvas && input && c.aspect !== "input") {
    const off = Math.abs(input.ratio / (width / height) - 1);
    if (off > 0.03) {
      aspectMessage = `The selected input is ${input.width}×${input.height} (${input.ratio.toFixed(2)}:1) but the canvas is ${(width / height).toFixed(2)}:1 — it will be cropped. Pick "Match input" to follow it.`;
    }
  }
  $("aspectWarn").hidden = !aspectMessage;
  $("aspectWarn").textContent = aspectMessage;
  for (const b of $("sizePresets").children) b.classList.toggle("on", +b.dataset.w === width && +b.dataset.h === height);
}

function renderDuration() {
  const { frames, fps } = S.common;
  const slider = $("frameSlider");
  const maxFrames = Number(slider.max) * 8 + 1;
  const marks = [1, 2, 5, 10, 15, 20, 30].filter((sec) => sec * fps >= 9 && sec * fps <= maxFrames);
  const shown = marks.length > 4 ? marks.filter((_, i) => i % 2 === marks.length % 2) : marks;
  $("durationMarkers").replaceChildren(...shown.map((sec) => el("span", {
    style: { left: `${(100 * ((sec * fps - 1) / 8 - Number(slider.min))) / (Number(slider.max) - Number(slider.min))}%` },
    text: `${sec}s`,
  })));
  slider.value = String((frames - 1) / 8);
  $("frameCount").textContent = frames;
  $("frameSecs").textContent = `${(frames / fps).toFixed(2)} s`;
  $("autoDurationRange").hidden = !S.common.autoDuration;
  $("frameSlider").disabled = S.common.autoDuration && TASKS[S.taskId].blocks.duration === "auto";
}

function syncCommonInputs() {
  const c = S.common;
  $("prompt").value = c.prompt;
  $("promptCount").textContent = c.prompt ? `${c.prompt.trim().split(/\s+/).length} words` : "";
  $("width").value = c.width;
  $("height").value = c.height;
  $("fps").value = c.fps;
  $("autoDuration").checked = c.autoDuration;
  $("autoMin").value = c.autoMin;
  $("autoMax").value = c.autoMax;
  $("seed").value = c.seed;
  $("quantize").value = c.quantize;
  $("lowRam").checked = c.lowRam;
  $("tileFrames").value = c.tileFrames;
  $("tileSpatial").value = c.tileSpatial;
  $("tileOverlap").value = c.tileOverlap;
  $("extraArgs").value = c.extraArgs;
  $("takeName").value = c.takeName;
  renderDuration();
}

// ── building the request ─────────────────────────────────────────────────

function collectErrors(task, v) {
  const errors = [];
  const blocks = task.blocks || {};
  if (blocks.prompt === "required" && !S.common.prompt.trim()) errors.push("Prompt is required.");
  const checkFields = (fields, values, prefix = "") => {
    for (const f of fields) {
      if (!visible(f, taskValues())) continue;
      if (f.type === "rows") {
        const rows = values[f.key] || [];
        if (rows.length < (f.minRows || 0)) errors.push(`${f.label}: add at least ${f.minRows}.`);
        rows.forEach((r, i) => checkFields(f.itemFields, r, `${f.label} ${i + 1} · `));
      } else if (f.required && (values[f.key] === undefined || values[f.key] === "" || values[f.key] === null)) {
        errors.push(`${prefix}${f.label} is required.`);
      } else if (f.type === "media" && values[f.key] && !inputByName(values[f.key])) {
        errors.push(`${prefix}${f.label}: "${values[f.key]}" is no longer in this session's inputs.`);
      }
    }
  };
  checkFields(task.fields, v);
  if (blocks.canvas && (S.common.width % 32 || S.common.height % 32)) errors.push("Width and height must be multiples of 32.");
  const reason = S.model.configured ? task.requires(v, S.model, requiresContext(task)) : "No model configured.";
  if (reason) errors.push(reason);
  return errors;
}

function buildRequest(seed) {
  const task = TASKS[S.taskId];
  const v = taskValues();
  const c = S.common;
  const blocks = task.blocks || {};
  const ctx = { frames: c.frames, fps: c.fps, width: c.width, height: c.height, seed, autoDuration: c.autoDuration, inputMeta };
  const args = [];
  if (blocks.prompt && c.prompt.trim()) args.push("--prompt", c.prompt.trim());
  if (blocks.canvas) args.push("--width", String(c.width), "--height", String(c.height));
  if (blocks.duration) {
    if (blocks.duration === "auto" && c.autoDuration) args.push("--auto-duration", `${c.autoMin}:${c.autoMax}`);
    else args.push("--frames", String(c.frames));
    args.push("--frame-rate", String(c.fps));
  }
  if (blocks.seed) args.push("--seed", String(seed));
  args.push(...task.build(v, ctx));
  if (blocks.lowRam && c.lowRam) args.push("--low-ram");
  if (blocks.tiling) {
    if (c.tileFrames > 1) args.push("--tile-frames", String(c.tileFrames));
    if (c.tileSpatial > 1) args.push("--tile-spatial", String(c.tileSpatial));
    if (c.tileFrames > 1 || c.tileSpatial > 1) args.push("--tile-overlap", String(c.tileOverlap));
  }
  args.push(...splitArgs(c.extraArgs));
  return {
    session: S.session,
    task_id: task.id,
    label: task.label,
    subcommand: task.cmd,
    args,
    output: task.output,
    quantize: blocks.quantize ? c.quantize : undefined,
    preview: previewRequest(),
    take_name: c.takeName || task.id,
    seed: blocks.seed ? seed : null,
    params: { ...snapshot(), seed },
  };
}

function refreshPreview() {
  const task = TASKS[S.taskId];
  let req;
  try { req = buildRequest(S.common.seed); } catch (e) { $("cmdPreview").textContent = String(e); return; }
  const shown = req.args.map((a) => {
    const text = typeof a === "object" ? (a.input ? `inputs/${a.input || "?"}` : a.path) : a;
    return /^[\w./:=@+,-]+$/.test(text) ? text : `'${String(text).replace(/'/g, "'\\''")}'`;
  });
  const tail = [];
  if (task.cmd !== "enhance" && task.cmd !== "slice" && task.cmd !== "train") tail.push("--model <model>");
  if (req.quantize) tail.push(`--quantize-on-load ${req.quantize}`);
  if (req.preview) {
    tail.push(`--stepwise-image-output-dir <session previews> --stepwise-interval ${req.preview.interval} --stepwise-frames ${req.preview.frames}`);
    if (req.preview.frame !== null) tail.push(`--stepwise-frame ${req.preview.frame}`);
  }
  if (task.output !== "none") tail.push("--output <session outputs>");
  $("cmdPreview").textContent = ["ltx-2-mlx", task.cmd, ...shown, ...tail].join(" ");
  const errors = collectErrors(task, taskValues());
  $("renderBtn").disabled = errors.length > 0;
  $("queueSeedsBtn").disabled = errors.length > 0 || !(task.blocks && task.blocks.seed);
  $("errors").hidden = true;
  renderAvailability();  // requires() can depend on non-structural values (frames, generated keyframes)
  refreshEstimate(req.params);
}

/** Pre-render estimate from this session's finished takes (median of matching takes). */
const refreshEstimate = debounce(async (params) => {
  if (!S.session) return;
  try {
    S.estimate = await api("/api/estimate", { session: S.session, params });
  } catch (e) {
    S.estimate = null;
  }
  const est = S.estimate || {};
  $("estimate").textContent = est.seconds ? `${est.exact ? "≈" : "~"} ${fmtSecs(est.seconds)}` : "";
  $("estimate").title = est.seconds
    ? `${est.exact ? "Median of" : "Scaled from"} ${est.samples} finished take${est.samples === 1 ? "" : "s"} in this session${est.exact ? " with the same settings" : " of this task"}`
    : "";
}, 400);

async function submit(count) {
  const task = TASKS[S.taskId];
  const errors = collectErrors(task, taskValues());
  $("errors").hidden = errors.length === 0;
  $("errors").textContent = errors.join("\n");
  if (errors.length) return;
  if (count > 1 && !(task.blocks && task.blocks.seed)) return;
  const seeds = count === 1 ? [S.common.seed] : Array.from({ length: count }, randomSeed);
  try {
    const { jobs } = await api("/api/render", { jobs: seeds.map((s) => buildRequest(s)) });
    if (count > 1) {
      trackCompareBatch(jobs);
      toast(`Queued ${jobs.length} seeds — the takes open side by side when they finish.`, { timeout: 4000 });
    }
    saveSettings.flush();
  } catch (e) {
    $("errors").hidden = false;
    $("errors").textContent = e.message;
  }
}

// ── settings persistence ─────────────────────────────────────────────────

function snapshot() {
  return { taskId: S.taskId, common: { ...S.common }, values: JSON.parse(JSON.stringify(S.values)) };
}

function restore(settings) {
  if (!settings || typeof settings !== "object") return;
  S.restoring = true;
  if (settings.taskId && TASKS[settings.taskId]) S.taskId = settings.taskId;
  if (settings.common) {
    const preview = { ...S.common.preview, ...(settings.common.preview || {}) };
    Object.assign(S.common, settings.common, { preview });
    if (!settings.common.aspect) setCanvasDims(S.common.width, S.common.height);
  }
  if (settings.values) S.values = { ...S.values, ...settings.values };
  syncCommonInputs();
  renderTask();
  S.restoring = false;
}

const saveSettings = debounce(() => {
  if (S.restoring || !S.session) return;
  api("/api/session/save", { session: S.session, settings: snapshot() }).catch(() => {});
}, 700);

// ── inputs library ───────────────────────────────────────────────────────

async function loadInputs() {
  S.inputs = await api(`/api/inputs?session=${encodeURIComponent(S.session)}`);
  renderLibrary();
  renderTask();
}

function renderLibrary() {
  $("inputCount").textContent = S.inputs.length ? String(S.inputs.length) : "";
  const kinds = ["image", "video", "audio"].filter((k) => S.inputs.some((i) => i.kind === k));
  const showFilter = S.inputs.length > 8 && kinds.length > 1;
  if (!showFilter || (S.inputFilter !== "all" && !kinds.includes(S.inputFilter))) S.inputFilter = "all";
  $("inputFilter").hidden = !showFilter;
  $("inputFilter").replaceChildren(...["all", ...kinds].map((k) => el("button", {
    type: "button", class: S.inputFilter === k ? "on" : "", "aria-pressed": String(S.inputFilter === k),
    text: k === "all" ? "All" : `${k[0].toUpperCase()}${k.slice(1)}s`,
    onclick: () => { S.inputFilter = k; renderLibrary(); },
  })));
  const shown = S.inputFilter === "all" ? S.inputs : S.inputs.filter((i) => i.kind === S.inputFilter);
  $("library").replaceChildren(...shown.map((i) => {
    let thumb = el("div", { class: "glyph", text: i.name });
    if (i.kind === "image") thumb = el("img", { src: i.url, alt: i.name, loading: "lazy" });
    else if (i.kind === "video") thumb = el("img", { src: i.thumb, alt: i.name, loading: "lazy" });
    else if (i.kind === "audio") thumb = el("div", { class: "glyph", text: `♪ ${i.name}` });
    const p = i.probe || {};
    const badge = i.kind === "video" || i.kind === "audio" ? (p.duration ? `${p.duration.toFixed(1)}s` : i.kind) : p.width ? `${p.width}×${p.height}` : "";
    return el("figure", { title: describeInput(i), onclick: () => fillSlot(i) },
      el("div", { class: "thumb" }, thumb),
      badge ? el("span", { class: "badge", text: badge }) : null,
      el("figcaption", { text: i.name }),
      el("button", { class: "delete-file", type: "button", title: "Delete input", text: "×",
        onclick: async (e) => {
          e.stopPropagation();
          if (!confirm(`Permanently delete this input from the session?\n${i.name}\n\nTasks that use it will need another input. This cannot be undone.`)) return;
          await api("/api/inputs/delete", { session: S.session, name: i.name });
          loadInputs();
        } }));
  }));
}

function fillSlot(input) {
  const task = TASKS[S.taskId];
  const v = taskValues();
  const tryFields = (fields, values) => {
    for (const f of fields) {
      if (!visible(f, v)) continue;
      if (f.type === "media" && (f.accept === input.kind || f.accept === "file") && !values[f.key]) { values[f.key] = input.name; return true; }
      if (f.type === "rows") {
        for (const row of values[f.key] || []) if (tryFields(f.itemFields, row)) return true;
      }
    }
    return false;
  };
  if (!tryFields(task.fields, v)) {
    const rowsField = task.fields.find((f) => f.type === "rows" && f.itemFields.some((i) => i.type === "media" && i.accept === input.kind));
    if (rowsField) {
      const row = rowDefaults(rowsField);
      row[rowsField.itemFields.find((i) => i.type === "media").key] = input.name;
      (v[rowsField.key] ||= []).push(row);
    } else {
      toast(`${task.label} has no empty ${input.kind} slot.`, { timeout: 4000 });
      return;
    }
  }
  renderTask();
  saveSettings();
}

async function uploadFiles(files) {
  const added = [];
  for (const file of files) {
    try {
      const res = await fetch(`/api/upload?session=${encodeURIComponent(S.session)}`, {
        method: "POST", body: file, headers: { "X-Filename": encodeURIComponent(file.name) },
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `upload failed (${res.status})`);
      added.push(data);
    } catch (e) {
      toast(`${file.name}: ${e.message}`, { kind: "error" });
    }
  }
  await loadInputs();
  for (const a of added) { const item = inputByName(a.name); if (item) fillSlot(item); }
}

// ── queue, progress, terminal ────────────────────────────────────────────

const STAGES = [["encode", "Encode"], ["load", "Load"], ["denoise", "Denoise"], ["decode", "Decode"], ["save", "Save"]];
let elapsedTimer = null;

const isActive = (job) => job.status === "running" || job.status === "cancelling";

function renderQueue(items) {
  S.queue = items;
  const running = items.find(isActive);
  const pending = items.filter((j) => j.status === "queued");
  if (running && S.runningId !== running.id) {
    S.runningId = running.id;
    S.followPreview = true;
    S.livePreview = null;
  }
  if (!running) S.runningId = null;
  $("lamp").classList.toggle("busy", !!running);
  $("lampText").textContent = running ? (pending.length ? `busy · ${pending.length} queued` : "busy") : pending.length ? `${pending.length} queued` : "idle";
  $("running").hidden = !running;
  clearInterval(elapsedTimer);
  if (running) {
    renderRunning(running);
    elapsedTimer = setInterval(renderRunningFromQueue, 1000);
  } else {
    document.title = "ltx studio";
  }
  $("queueWrap").hidden = pending.length === 0;
  $("queueList").replaceChildren(...pending.map((j) => el("li", {},
    el("span", { text: `${j.label}${j.seed !== null && j.seed !== undefined ? ` · seed ${j.seed}` : ""}${j.estimate_s ? ` · ≈ ${fmtSecs(j.estimate_s)}` : ""} · queued` }),
    el("button", { class: "ghost sm", type: "button", text: "Remove",
      onclick: () => api("/api/cancel", { id: j.id }).catch((err) => toast(err.message, { kind: "error" })) }))));
}

function renderRunningFromQueue() {
  const running = S.queue.find(isActive);
  if (running) renderRunning(running);
}

/** Rough overall progress for the tab title: stage bands, with denoising weighted most.
 *  The number of denoising stages isn't known up front, so stage 1 gets most of the band and later
 *  stages (the short two-stage refine) the rest. */
function overallPercent(job) {
  const p = job.progress || {};
  const denoise = (p.stage || 1) > 1 ? [75, 88] : [15, 75];
  const bands = { encode: [0, 8], load: [8, 15], denoise, decode: [88, 98], save: [98, 100] };
  const [lo, hi] = bands[p.stage_key] || [0, 0];
  return Math.round(lo + (p.stage_key === "denoise" && p.total ? (hi - lo) * Math.min(1, p.step / p.total) : 0));
}

function renderRunning(job) {
  const p = job.progress || {};
  const stageIndex = STAGES.findIndex(([key]) => key === p.stage_key);
  $("stepper").replaceChildren(...STAGES.map(([key, label], i) => el("li", {
    class: i < stageIndex ? "done" : i === stageIndex ? "active" : "",
  }, key === "denoise" && i === stageIndex && p.stage > 1 ? `${label} · stage ${p.stage}` : label)));
  const pct = p.total ? Math.min(100, Math.round((100 * p.step) / p.total)) : null;
  const cancelling = job.status === "cancelling";
  $("phaseName").textContent = `${job.label} — ${cancelling ? "stopping…" : p.phase || "starting"}`;
  $("phasePct").textContent = p.total ? `${p.step}/${p.total}` : "";
  $("phaseBar").style.width = pct === null ? "" : `${pct}%`;
  $("phaseBar").parentElement.classList.toggle("indeterminate", pct === null);
  const started = job.started ? Date.parse(job.started) : null;
  const elapsed = started ? (Date.now() - started) / 1000 : 0;
  $("elapsed").textContent = started ? `${fmtSecs(elapsed)} elapsed${p.note ? ` · ${p.note}` : ""}` : "";
  let eta = "";
  if (p.stage_key === "denoise" && p.eta_s) {
    const left = p.eta_s - (Date.now() / 1000 - p.eta_at);
    eta = left >= 1.5 ? `≈ ${fmtSecs(left)} left in this denoising stage` : "denoising stage almost done";
  } else if (job.estimate_s && job.estimate_s > elapsed) {
    eta = `≈ ${fmtSecs(job.estimate_s - elapsed)} left (estimate from earlier takes)`;
  }
  $("eta").textContent = eta;
  $("stopBtn").disabled = cancelling;
  $("stopBtn").textContent = cancelling ? "Stopping…" : "Stop";
  $("stopBtn").onclick = () => api("/api/cancel", { id: job.id }).catch((err) => toast(err.message, { kind: "error" }));
  const latest = job.preview_latest || (S.livePreview && S.livePreview.id === job.id ? S.livePreview : null);
  $("followPreviewBtn").hidden = !latest || S.followPreview;
  $("followPreviewBtn").onclick = () => {
    S.followPreview = true;
    S.selectedTake = null;
    S.selectedTimeline = null;
    markSelection();
    showLivePreview({ ...latest, id: job.id, count: job.preview_count || latest.count });
    $("followPreviewBtn").hidden = true;
  };
  document.title = `(${overallPercent(job)}%) ltx studio`;
}

function onJobEvent(job) {
  const idx = S.queue.findIndex((j) => j.id === job.id);
  if (idx >= 0) { S.queue[idx] = job; renderQueue(S.queue); }
  if (job.status === "running" && job.session === S.session && job.helm_job) showTerminalJob(job.helm_job);
  noteBatchJob(job);
  if (job.status === "failed") {
    toast(`${job.label} failed: ${job.error || "unknown error"}`, { kind: "error", hint: job.hint || "", timeout: 15000 });
    notify(`Render failed: ${job.label}`, job.error || "");
  } else if (job.status === "done") {
    if (job.session !== S.session) toast(`${job.label} finished in session ${job.session}.`, { kind: "ok" });
    notify(`Render finished: ${job.label}`, job.elapsed ? `in ${fmtSecs(job.elapsed)}` : "");
  }
}

function notificationsOn() {
  return S.preferences.notify === true && "Notification" in window && Notification.permission === "granted";
}

function notify(title, body) {
  if (!notificationsOn() || !document.hidden) return;
  try { new Notification(title, { body }); } catch (e) { /* notifications unavailable */ }
}

function renderNotifyButton() {
  const on = notificationsOn();
  $("notifyButton").classList.toggle("on", on);
  $("notifyButton").setAttribute("aria-pressed", String(on));
  $("notifyButton").title = on ? "Notifications on — click to turn off" : "Notify me when a render finishes while this tab is in the background";
}

async function toggleNotify() {
  if (!("Notification" in window)) { toast("This browser doesn't support notifications."); return; }
  if (S.preferences.notify === true) {
    savePreference("notify", false);
  } else {
    const permission = Notification.permission === "granted" ? "granted" : await Notification.requestPermission();
    if (permission !== "granted") { toast("Notifications are blocked for this page in the browser settings."); return; }
    savePreference("notify", true);
    toast("You'll get a notification when a render finishes while this tab is in the background.", { kind: "ok", timeout: 4000 });
  }
  renderNotifyButton();
}

/** The terminal is helm-terminal, streaming one render's log from helmstudio, which keeps it. */
function showTerminalJob(job) {
  const terminal = $("terminal");
  if (!job) terminal.removeAttribute("job");
  else if (terminal.getAttribute("job") !== job) terminal.setAttribute("job", job);
}

/** Show the log of the session's latest render; a render of the session that starts later takes its place. */
async function loadTerminal() {
  try {
    const { job } = await api(`/api/terminal?session=${encodeURIComponent(S.session)}`);
    showTerminalJob(job);
  } catch (e) { /* the log is a convenience */ }
}

function connectEvents() {
  const es = new EventSource("/api/events");
  es.onopen = () => renderQueue(S.queue);
  es.onmessage = (event) => {
    const { type, data } = JSON.parse(event.data);
    if (type === "queue") renderQueue(data);
    else if (type === "progress") {
      const job = S.queue.find((j) => j.id === data.id);
      if (job) { job.progress = data.progress; renderRunning(job); }
    } else if (type === "job") onJobEvent(data);
    else if (type === "takes" && data.session === S.session) loadTakes(true);
    else if (type === "preview" && data.session === S.session) {
      const job = S.queue.find((j) => j.id === data.id);
      if (job) { job.preview_latest = data; job.preview_count = data.count; renderRunning(job); }
      showLivePreview(data);
    }
  };
  es.onerror = () => { $("lampText").textContent = "reconnecting…"; };
}

/** Drag the terminal's top edge to resize it; the height is one of the page's preferences. */
function bindTerminalResize() {
  const box = $("consoleContainer");
  let startY = 0, startHeight = 0, dragging = false;
  const handle = $("resizeHandle");
  handle.addEventListener("pointerdown", (e) => {
    dragging = true; startY = e.clientY; startHeight = box.offsetHeight;
    handle.setPointerCapture(e.pointerId);
    document.body.classList.add("resizing");
  });
  handle.addEventListener("pointermove", (e) => {
    if (dragging) box.style.height = `${Math.max(120, Math.min(startHeight + (startY - e.clientY), window.innerHeight * 0.75))}px`;
  });
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("resizing");
    savePreference("terminalHeight", box.offsetHeight);
  };
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);
}

// ── prompt history ───────────────────────────────────────────────────────

/** Unique prompts from this session's takes, newest first. */
function promptHistory() {
  const seen = new Set();
  const items = [];
  for (const t of S.takes) {
    const text = takePrompt(t).trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    items.push({ text, take: t.name });
  }
  return items;
}

function renderHistoryButton() {
  $("historyBtn").disabled = promptHistory().length === 0;
}

function renderHistoryMenu() {
  const items = promptHistory();
  $("historyMenu").replaceChildren(...(items.length ? items.map((item) => el("button", {
    type: "button", role: "menuitem", title: item.text,
    onclick: (e) => { e.stopPropagation(); closePopmenus(); setPrompt(item.text); },
  }, el("span", { class: "history-text", text: item.text }), el("small", { text: item.take })))
    : [el("div", { class: "popmenu-empty", text: "No prompts in this session's takes yet." })]));
}

function setPrompt(text) {
  S.common.prompt = text;
  $("prompt").value = text;
  $("promptCount").textContent = text ? `${text.trim().split(/\s+/).length} words` : "";
  refreshPreview();
  saveSettings();
  $("prompt").focus();
}

// ── sessions + model ─────────────────────────────────────────────────────

async function activateSession(name) {
  const res = await api("/api/session/activate", { session: name });
  closePopmenus();
  S.session = res.name;
  S.values = {};
  S.selectedTake = null;
  S.selectedTimeline = null;
  S.compare = [];
  const cfg = await api("/api/sessions");
  S.sessions = cfg.sessions;
  renderSessions();
  showInViewer(null);
  await Promise.all([loadInputs(), loadTakes(), loadTimeline(), loadTerminal()]);
  restore(res.settings);
  if (!res.settings || !res.settings.taskId) { syncCommonInputs(); renderTask(); }
}

function renderSessions() {
  $("sessionSelect").replaceChildren(...S.sessions.map((s) => el("option", { value: s, text: s })));
  $("sessionSelect").value = S.session;
}

function openSessionModal(mode) {
  $("sessionModalTitle").textContent = mode === "duplicate" ? `Duplicate ${S.session}` : "New session";
  $("confirmSession").textContent = mode === "duplicate" ? "Duplicate" : "Create";
  let n = S.sessions.length + 1;
  while (S.sessions.includes(`session-${n}`)) n += 1;
  $("sessionNameInput").value = mode === "duplicate" ? `${S.session}-copy` : `session-${n}`;
  $("sessionError").hidden = true;
  $("sessionModal").hidden = false;
  $("sessionNameInput").focus();
  $("sessionNameInput").select();
  $("confirmSession").onclick = async () => {
    const name = $("sessionNameInput").value.trim();
    try {
      if (!name) throw new Error("Enter a name.");
      if (S.sessions.includes(name)) throw new Error("That session already exists.");
      if (mode === "duplicate") await api("/api/session/duplicate", { session: S.session, new_name: name });
      $("sessionModal").hidden = true;
      await activateSession(name);
    } catch (e) {
      $("sessionError").hidden = false;
      $("sessionError").textContent = e.message;
    }
  };
}

function renderModelCaps(info) {
  const marks = { ok: "✓", warn: "!", bad: "✕", info: "i" };
  $("modelChecks").replaceChildren(...(info.checks || []).map((c) => el("li", { class: c.level },
    el("b", { text: marks[c.level] || "·" }), el("span", { text: c.label }), el("code", { text: c.detail }))));
  const row = (label, ok, text) => [el("b", { text: label }), el("span", { class: ok === null ? "" : ok ? "yes" : "no", text })];
  if (!info.configured) {
    $("modelCaps").replaceChildren();
    return;
  }
  $("modelCaps").replaceChildren(
    ...row("model", null, info.model),
    ...row("type", null, info.local ? (info.is_25 ? "LTX-2.5" : "LTX-2.3 / other") : "Hugging Face repo (not inspected)"),
    ...row("distilled", info.has_distilled, info.has_distilled ? "yes" : "no"),
    ...row("dev", info.has_dev, info.has_dev ? "yes — two-stage, a2v, retake, extend, keyframe" : "no — dev-model tasks unavailable"),
    ...row("IC-LoRA", !info.is_25, info.is_25 ? "not on LTX-2.5 packs" : "available"),
  );
}

function applyModel(info) {
  S.model = info;
  $("modelButton").classList.toggle("warn-on", !info.configured);
  $("modelButtonText").textContent = info.configured ? `Model · ${info.model.split("/").filter(Boolean).pop()}` : "Set model";
  $("modelDot").className = `dot ${info.status || ""}`;
  const problems = (info.checks || []).filter((c) => c.level === "bad" || c.level === "warn").map((c) => `${c.label}: ${c.detail}`);
  $("modelButton").title = problems.length ? problems.join("\n") : "Model and setup";
  renderModelCaps(info);
  renderTask();
}

// ── keyboard ─────────────────────────────────────────────────────────────

function onGlobalKeydown(e) {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key === "Enter") {
    e.preventDefault();
    if ($("sessionModal").hidden && $("modelModal").hidden) submit(e.shiftKey ? 3 : 1);
    return;
  }
  if (e.key === "Escape") {
    let closed = false;
    document.querySelectorAll(".modal").forEach((modal) => { if (!modal.hidden) { modal.hidden = true; closed = true; } });
    if (!$("compare").hidden) { exitCompare(); closed = true; }
    if ([...document.querySelectorAll(".popmenu")].some((m) => !m.hidden)) { closePopmenus(); closed = true; }
    if (closed) e.preventDefault();
    return;
  }
  if (mod || e.altKey || isTyping(e.target) || e.target.tagName === "BUTTON" || e.target.tagName === "A") return;
  if ([...document.querySelectorAll(".modal")].some((modal) => !modal.hidden)) return;
  const player = $("player");
  const hasVideo = player.classList.contains("on") && player.currentSrc;
  if (e.key === " " && hasVideo) {
    e.preventDefault();
    if (player.paused) player.play().catch(() => {}); else player.pause();
  } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && hasVideo) {
    e.preventDefault();
    player.pause();
    const take = S.takes.find((t) => t.name === S.selectedTake);
    const fps = (take && take.probe && take.probe.fps) || S.common.fps || 24;
    player.currentTime = Math.max(0, player.currentTime + (e.key === "ArrowRight" ? 1 : -1) / fps);
  } else if (e.key === "j" || e.key === "k") {
    const names = visibleTakes().map((t) => t.name);
    if (!names.length) return;
    const index = names.indexOf(S.selectedTake);
    const next = index < 0 ? 0 : Math.min(names.length - 1, Math.max(0, index + (e.key === "j" ? 1 : -1)));
    showSidePanel("takes");
    selectTake(names[next]);
    const li = document.querySelector(`#takeList > li[data-name="${CSS.escape(names[next])}"]`);
    if (li) li.scrollIntoView({ block: "nearest" });
  }
}

// ── helmstudio: its components, the theme, the gallery and the timeline ──
//
// ltx studio runs only under helmstudio or `helm dev`, and all of this comes
// from helmstudio through the proxy server.py mounts at /helm/ — the browser
// runtime and the components, as index.html links helm-css's tokens and this
// studio's hue — so nothing of helmstudio's is copied into ltx studio, and the
// page never holds the token. It
//   - follows helmstudio's theme, which is the page's only theme;
//   - gives the terminal (helm-terminal) the page's client;
//   - shows the gallery of the takes this studio recorded (helm-gallery);
//   - opens Create Timeline on helmstudio's timeline: a sequence helmstudio
//     keeps, edited in helm-timeline — reorder, trim, dissolve, gain, undo —
//     and exported from it, with the gallery as its picker.
//
// There is one helm-gallery on the page, shared by browsing and picking: each
// gallery holds an event stream open, as the terminal does while a render
// runs, and a page has six connections to its host.

const HELM_SDK = "/helm/sdk/v1";
// The frame rates a sequence can have, of which a new one takes the nearest to its first take's.
const SEQUENCE_RATES = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];

function helmDialog(className, html) {
  const d = document.createElement("dialog");
  d.className = `helmstudio-dialog ${className}`;
  d.innerHTML = html;
  document.body.append(d);
  return d;
}

// ── the gallery: browsing, and the timeline's picker ─────────────────────

function galleryDialog() {
  const d = helmDialog("helmstudio-gallery", `
    <div class="helmstudio-head">
      <h2 class="helmstudio-title">Gallery</h2>
      <span class="helmstudio-status" role="status"></span>
      <span class="helmstudio-spacer"></span>
      <button class="ghost sm" type="button" data-act="close">Close</button>
    </div>
    <helm-gallery scope="self" kind="video"></helm-gallery>`);
  const gallery = d.querySelector("helm-gallery");
  const title = d.querySelector(".helmstudio-title");
  const status = d.querySelector(".helmstudio-status");
  let choose = null;

  const finish = (item) => {
    if (!choose) return;
    const resolve = choose;
    choose = null;
    gallery.removeAttribute("picker");
    title.textContent = "Gallery";
    status.textContent = "";
    resolve(item);
  };
  gallery.addEventListener("pick", (event) => {
    const { item } = event.detail;
    finish(item);
    d.close();
  });
  d.addEventListener("close", () => finish(null));
  d.querySelector('[data-act="close"]').addEventListener("click", () => d.close());

  return {
    browse() {
      d.showModal();
    },
    /** Opens the gallery as a picker; resolves to the chosen item, or null. */
    pick(purpose) {
      finish(null);
      title.textContent = purpose;
      status.textContent = "Select a take, then Use this.";
      gallery.setAttribute("picker", "");
      d.showModal();
      return new Promise((resolve) => {
        choose = resolve;
      });
    },
  };
}

// ── the timeline ─────────────────────────────────────────────────────────

function timelineDialog(helm, picker) {
  const d = helmDialog("helmstudio-timeline", `
    <div class="helmstudio-head">
      <h2 class="helmstudio-title">Timeline</h2>
      <span class="helmstudio-status" role="status"></span>
      <span class="helmstudio-spacer"></span>
      <button class="ghost sm" type="button" data-act="new">New sequence</button>
      <button class="ghost sm" type="button" data-act="close">Close</button>
    </div>
    <p class="helmstudio-empty" hidden>No sequence yet. New sequence starts one from a take.</p>
    <helm-timeline chooser editable hidden></helm-timeline>`);
  // `chooser` is the editor's own list of the sequences this studio may read
  // (helm-ui-sdk, 04 §11). It replaces the <select> this dialog kept beside
  // it, and picking one is the component's business now.
  const tl = d.querySelector("helm-timeline");
  const empty = d.querySelector(".helmstudio-empty");
  const status = d.querySelector(".helmstudio-status");
  const say = (text) => {
    status.textContent = text;
  };

  // A clip is labelled by the take that made it: its task and its seed.
  const labels = new Map();
  async function learnLabels() {
    let cursor = null;
    do {
      const page = await helm.gallery.query({ scope: "self", kind: "video", limit: 200, cursor });
      for (const item of page.items || []) {
        const seed = item.params && item.params.seed != null ? ` · seed ${item.params.seed}` : "";
        labels.set(item.asset_id, `${item.title || "take"}${seed}`);
      }
      cursor = page.next_cursor || null;
    } while (cursor);
  }
  tl.labelFor = (clip) => labels.get(clip.asset_id) || "";

  // What is left to this page is what the component cannot know: which take
  // made a clip. The editor finds the sequences and opens one; naming a
  // sequence is only asking it to open that one instead of the first.
  async function load(selectId) {
    const page = await helm.timeline.list({ limit: 50 });
    const sequences = page.items || [];
    empty.hidden = sequences.length > 0;
    tl.hidden = sequences.length === 0;
    if (!sequences.length) return;
    await learnLabels().catch(() => {});
    const chosen = sequences.find((s) => s.id === selectId) || sequences[0];
    if (tl.getAttribute("timeline") !== chosen.id) tl.setAttribute("timeline", chosen.id);
  }

  tl.addEventListener("exported", () => {
    say("Exported — the sequence is in the gallery.");
    loadTimeline(); // the Timeline tab lists exported sequences (takes.js)
  });

  // What goes on a sequence is this page's to choose: the editor asks, the
  // gallery answers.
  tl.addEventListener("add-request", async () => {
    const item = await picker.pick("Add a take to the sequence");
    if (item) await tl.append(item.asset_id);
  });

  d.querySelector('[data-act="new"]').addEventListener("click", async () => {
    const item = await picker.pick("Start a sequence from a take");
    if (!item) return;
    const asset = item.asset || {};
    if (!asset.width || !asset.height) {
      say("That take has no dimensions to build a sequence at.");
      return;
    }
    const want = asset.fps || 24;
    const fps = SEQUENCE_RATES.reduce((best, rate) => (Math.abs(rate - want) < Math.abs(best - want) ? rate : best));
    try {
      const made = await helm.timeline.create({
        name: `ltx sequence ${new Date().toLocaleString()}`,
        target: { width: asset.width, height: asset.height, fps },
        clips: [{ asset_id: item.asset_id }],
      });
      say("");
      await load(made.id);
    } catch (err) {
      say(`The sequence could not be made: ${err.message}`);
    }
  });
  d.querySelector('[data-act="close"]').addEventListener("click", () => d.close());

  return {
    /** open shows the editor, on `selectId` when one is named. */
    async open(selectId) {
      d.showModal();
      say("");
      try {
        await load(selectId);
      } catch (err) {
        empty.hidden = false;
        empty.textContent = `The sequences could not be read: ${err.message}`;
      }
    },
  };
}

/** Connect to helmstudio: its runtime and components, the theme, the page's client, and the dialogs. */
async function connectHelmstudio() {
  const { connect, themeBridge } = await import(`${HELM_SDK}/helm-runtime.js`);
  await import(`${HELM_SDK}/helm-ui.js`);
  themeBridge();
  const helm = (window.helm = connect());
  // The terminal may have been given its render before there was a client.
  $("terminal").client = helm;

  const gallery = galleryDialog();
  const timeline = timelineDialog(helm, gallery);

  const galleryButton = $("helmGalleryButton");
  galleryButton.hidden = false;
  galleryButton.addEventListener("click", () => gallery.browse());

  $("timelineButton").addEventListener("click", () => timeline.open());

  // The Timeline panel lists helmstudio's sequences and knows nothing about
  // this dialog; it says which one was asked for, and this decides what that
  // means. With nothing behind the proxy nothing listens, and nothing lists a
  // sequence to ask about either.
  document.addEventListener("ltx:open-sequence", (event) => {
    timeline.open(event.detail && event.detail.id);
  });
}

// ── wiring ───────────────────────────────────────────────────────────────

function bindCommon() {
  const c = S.common;
  const bindNum = (id, key, after) => $(id).addEventListener("input", (e) => {
    const value = num(e.target.value);
    if (value !== null) { c[key] = value; if (after) after(); refreshPreview(); saveSettings(); }
  });
  $("prompt").addEventListener("input", (e) => {
    c.prompt = e.target.value;
    $("promptCount").textContent = c.prompt ? `${c.prompt.trim().split(/\s+/).length} words` : "";
    refreshPreview();
    saveSettings();
  });
  $("clearPrompt").addEventListener("click", () => { if (!c.prompt || confirm("Clear the prompt?")) setPrompt(""); });
  const historyWrap = $("historyBtn").parentElement;
  $("historyBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    const menu = $("historyMenu");
    const open = menu.hidden;
    closePopmenus(menu);
    if (open) renderHistoryMenu();
    menu.hidden = !open;
    $("historyBtn").setAttribute("aria-expanded", String(open));
  });
  historyWrap.addEventListener("click", (e) => e.stopPropagation());
  bindNum("width", "width", renderCanvas);
  bindNum("height", "height", renderCanvas);
  for (const id of ["width", "height"]) {
    $(id).addEventListener("change", () => {
      const grid = currentGrid();
      setCanvasDims(snapDimension(c.width, grid), snapDimension(c.height, grid));
      renderCanvas(); refreshPreview(); saveSettings();
    });
  }
  $("megapixels").addEventListener("input", (e) => {
    c.megapixels = clampMegapixels(Number(e.target.value));
    solveCurrentCanvas();
    renderCanvas(); refreshPreview(); saveSettings();
  });
  $("aspectSelect").addEventListener("change", (e) => {
    c.aspect = e.target.value;
    // Custom keeps the current size and locks its ratio for the megapixel slider.
    if (c.aspect === "custom") c.customRatio = c.width / c.height;
    else solveCurrentCanvas();
    renderCanvas(); refreshPreview(); saveSettings();
  });
  bindNum("fps", "fps", renderDuration);
  bindNum("seed", "seed");
  bindNum("autoMin", "autoMin");
  bindNum("autoMax", "autoMax");
  bindNum("tileFrames", "tileFrames");
  bindNum("tileSpatial", "tileSpatial");
  bindNum("tileOverlap", "tileOverlap");
  $("frameSlider").addEventListener("input", (e) => { c.frames = Number(e.target.value) * 8 + 1; renderDuration(); refreshPreview(); saveSettings(); });
  $("autoDuration").addEventListener("change", (e) => { c.autoDuration = e.target.checked; renderDuration(); refreshPreview(); saveSettings(); });
  $("quantize").addEventListener("change", (e) => { c.quantize = e.target.value; refreshPreview(); saveSettings(); });
  $("lowRam").addEventListener("change", (e) => { c.lowRam = e.target.checked; refreshPreview(); saveSettings(); });
  $("extraArgs").addEventListener("input", (e) => { c.extraArgs = e.target.value; refreshPreview(); saveSettings(); });
  $("takeName").addEventListener("input", (e) => { c.takeName = e.target.value; saveSettings(); });
  const previewChanged = () => { renderPreviewOptions(); refreshPreview(); saveSettings(); };
  $("previewEnabled").addEventListener("change", (e) => { c.preview.enabled = e.target.checked; previewChanged(); });
  $("previewInterval").addEventListener("input", (e) => { const v = num(e.target.value); if (v !== null && v >= 1) { c.preview.interval = Math.round(v); previewChanged(); } });
  $("previewFrames").addEventListener("change", (e) => { c.preview.frames = Number(e.target.value); previewChanged(); });
  $("previewPosition").addEventListener("change", (e) => { c.preview.position = e.target.value; previewChanged(); });
  $("previewFrame").addEventListener("input", (e) => { const v = num(e.target.value); if (v !== null) { c.preview.frame = Math.round(v); previewChanged(); } });
  $("previewSlider").addEventListener("input", renderScrub);
  $("previewClose").addEventListener("click", () => { const take = S.scrub && S.scrub.take; hidePreview(); if (take) selectTake(take.name); });
  $("dice").addEventListener("click", () => { c.seed = randomSeed(); $("seed").value = c.seed; refreshPreview(); saveSettings(); });

  $("sizePresets").replaceChildren(...SIZE_PRESETS.map(([id, label, w, h]) => el("button", {
    type: "button", "data-w": String(w), "data-h": String(h), text: `${label} ${w}×${h}`,
    onclick: () => { setCanvasDims(w, h); renderCanvas(); refreshPreview(); saveSettings(); },
  })));
}

function showSidePanel(which, remember = true) {
  document.querySelectorAll(".sidetabs button").forEach((b) => {
    const on = b.dataset.side === which;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  $("takesPanel").hidden = which !== "takes";
  $("timelinePanel").hidden = which !== "timeline";
  if (remember) savePreference("side", which);
}

function bindChrome() {
  $("taskSelect").addEventListener("change", (e) => { S.taskId = e.target.value; $("errors").hidden = true; renderTask(); saveSettings(); });
  $("renderBtn").addEventListener("click", () => submit(1));
  $("queueSeedsBtn").addEventListener("click", () => submit(3));
  $("notifyButton").addEventListener("click", toggleNotify);

  document.querySelectorAll(".sidetabs button").forEach((b) => b.addEventListener("click", () => showSidePanel(b.dataset.side)));
  $("starFilter").addEventListener("change", renderTakes);
  $("compareGrid").addEventListener("click", () => openCompareGrid());
  $("compareWipe").addEventListener("click", () => openWipe());
  $("compareClear").addEventListener("click", () => { S.compare = []; renderTakes(); if (!$("compare").hidden) exitCompare(); });

  $("sessionSelect").addEventListener("change", (e) => activateSession(e.target.value));
  $("sessionMenuBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    const menu = $("sessionMenu");
    const open = menu.hidden;
    closePopmenus(menu);
    menu.hidden = !open;
    $("sessionMenuBtn").setAttribute("aria-expanded", String(open));
  });
  $("newSession").addEventListener("click", () => { closePopmenus(); openSessionModal("new"); });
  $("duplicateSession").addEventListener("click", () => { closePopmenus(); openSessionModal("duplicate"); });
  $("cancelSession").addEventListener("click", () => { $("sessionModal").hidden = true; });
  $("sessionNameInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("confirmSession").click(); });
  $("deleteSession").addEventListener("click", async () => {
    closePopmenus();
    if (!confirm(`Delete session "${S.session}" with its settings and inputs?\n\nIts takes stay in helmstudio's gallery.`)) return;
    try {
      const res = await api("/api/session/delete", { session: S.session });
      await activateSession(res.name);
    } catch (e) {
      toast(e.message, { kind: "error" });
    }
  });

  $("modelButton").addEventListener("click", () => {
    $("modelInput").value = S.model.model || "";
    $("gemmaInput").value = S.model.gemma || "";
    $("modelError").hidden = true;
    $("modelModal").hidden = false;
  });
  $("cancelModel").addEventListener("click", () => { $("modelModal").hidden = true; });
  $("recheckModel").addEventListener("click", async () => {
    try {
      applyModel(await api("/api/model/check", {}));
      toast("Setup rechecked.", { kind: "ok", timeout: 2500 });
    } catch (e) {
      toast(e.message, { kind: "error" });
    }
  });
  $("saveModel").addEventListener("click", async () => {
    try {
      const info = await api("/api/model", { model: $("modelInput").value, gemma: $("gemmaInput").value });
      applyModel(info);
      $("modelModal").hidden = true;
    } catch (e) {
      $("modelError").hidden = false;
      $("modelError").textContent = e.message;
    }
  });

  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); uploadFiles([...e.dataTransfer.files]); });
  $("fileInput").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });

  document.addEventListener("keydown", onGlobalKeydown);
  document.addEventListener("click", () => closePopmenus());
  bindTerminalResize();
}

/** Remember one of the page's preferences. helmstudio keeps them, like everything else ltx studio keeps. */
function savePreference(key, value) {
  S.preferences[key] = value;
  api("/api/preferences", { changes: { [key]: value } }).catch(() => {});
}

function applyPreferences(preferences) {
  S.preferences = preferences || {};
  showSidePanel(S.preferences.side === "timeline" ? "timeline" : "takes", false);
  if (S.preferences.terminalHeight >= 120) $("consoleContainer").style.height = `${S.preferences.terminalHeight}px`;
  renderNotifyButton();
}

async function init() {
  renderTaskSelect();
  bindCommon();
  bindChrome();
  syncCommonInputs();
  const cfg = await api("/api/config");
  applyPreferences(cfg.preferences);
  S.sessions = cfg.sessions;
  applyModel(cfg.model);
  connectEvents();
  await activateSession(cfg.active);
  if (!cfg.ffmpeg) {
    toast("ffmpeg is not on PATH — media probing and frame/audio extraction are disabled.", { kind: "error", hint: "Install it with `brew install ffmpeg` and restart the studio.", timeout: 0 });
  }
  if (!cfg.model.configured) $("modelButton").click();
}

init().catch((e) => { console.error(e); toast(`Failed to start: ${e.message}`, { kind: "error", timeout: 0 }); });
connectHelmstudio().catch((e) => { console.error(e); toast(`helmstudio's components did not load: ${e.message}`, { kind: "error", timeout: 0 }); });
