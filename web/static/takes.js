// ltx studio — viewer, takes and timeline lists, preview scrubber, seed grid and A/B wipe.
// Follows h3 studio's static/takes.js. Uses S, loadInputs, restore, … from app.js.

"use strict";

// ── viewer ───────────────────────────────────────────────────────────────

function setCaption(text) {
  $("viewerCaption").hidden = !text;
  $("viewerCaption").textContent = text || "";
  $("viewerCaption").title = text || "";  // the caption is clamped to three lines
}

function closeCompare() {
  const box = $("compare");
  box.querySelectorAll("video").forEach((v) => v.pause());
  box.replaceChildren();
  box.hidden = true;
  $("viewer").classList.remove("comparing");
}

/** Close the grid or wipe and go back to the selected take (or the empty viewer). */
function exitCompare() {
  closeCompare();
  if (S.selectedTake) selectTake(S.selectedTake, false);
  else if (S.selectedTimeline) selectTimeline(S.selectedTimeline);
  else showInViewer(null);
}

function hidePreview() {
  $("previewImg").hidden = true;
  $("previewImg").removeAttribute("src");
  $("previewBadge").hidden = true;
  $("previewScrub").hidden = true;
  S.scrub = null;
}

function showPreviewImage(url, badge) {
  closeCompare();
  const player = $("player");
  player.pause();
  player.classList.remove("on");
  $("viewerEmpty").hidden = true;
  $("previewImg").src = url;
  $("previewImg").hidden = false;
  $("previewBadge").textContent = badge;
  $("previewBadge").hidden = false;
}

function previewLabel(info) {
  const stage = info.stage ? ` · stage ${info.stage}` : "";
  return `Preview step ${info.step}/${info.total}${stage}`;
}

function showLivePreview(info) {
  S.livePreview = info;
  if (!S.followPreview) return;
  $("previewScrub").hidden = true;
  S.scrub = null;
  setCaption(`live preview · ${info.name}${info.count ? ` · ${info.count} so far` : ""}`);
  showPreviewImage(`${info.url}?t=${info.count || 0}`, previewLabel(info));
}

function openScrubber(take) {
  const list = take.previews || [];
  if (!list.length) return;
  S.followPreview = false;
  S.scrub = { take, list };
  const slider = $("previewSlider");
  slider.max = String(list.length - 1);
  slider.value = String(list.length - 1);
  renderScrub();
}

function renderScrub() {
  if (!S.scrub) return;
  const index = Number($("previewSlider").value);
  const path = S.scrub.list[index];
  const name = path.split("/").pop();
  const m = name.match(/_s(\d+)_step(\d+)of(\d+)\.webp$/) || name.match(/_step(\d+)of(\d+)\.webp$/);
  const info = m && m.length === 4 ? { stage: Number(m[1]), step: Number(m[2]), total: Number(m[3]) }
    : m ? { stage: 0, step: Number(m[1]), total: Number(m[2]) } : { stage: 0, step: 0, total: 0 };
  showPreviewImage(`/stage/${path}`, previewLabel(info));
  $("previewScrub").hidden = false;
  $("previewScrubLabel").textContent = `${index + 1}/${S.scrub.list.length}`;
  setCaption(`${S.scrub.take.name} · previews · ${name}`);
}

/** Play a take or an exported sequence in the viewer, or clear it when item is null. */
function showInViewer(item, caption = "", autoplay = false) {
  closeCompare();
  hidePreview();
  const player = $("player");
  if (item) {
    player.src = item.url;
    player.classList.add("on");
    $("viewerEmpty").hidden = true;
    setCaption(caption);
    if (autoplay) player.play().catch(() => {});
  } else {
    S.selectedTake = null;
    S.selectedTimeline = null;
    player.pause();
    player.removeAttribute("src");
    player.load();
    player.classList.remove("on");
    $("viewerEmpty").hidden = false;
    setCaption("");
  }
  markSelection();
}

function markSelection() {
  document.querySelectorAll("#takeList > li[data-name]").forEach((li) => li.classList.toggle("on", li.dataset.name === S.selectedTake));
  document.querySelectorAll("#timelineList > li[data-name]").forEach((li) => li.classList.toggle("on", li.dataset.name === S.selectedTimeline));
}

// ── takes list ───────────────────────────────────────────────────────────

async function loadTakes(selectNewest = false) {
  S.takes = await api(`/api/takes?session=${encodeURIComponent(S.session)}`);
  renderTakes();
  renderHistoryButton();
  // Never auto-select over a seed grid that is open or about to open.
  if (selectNewest && S.takes.length && $("compare").hidden && !S.batchOpening) selectTake(S.takes[0].name, false);
}

function takeMeta(t) {
  const p = t.probe || {};
  const bits = [];
  if (t.label) bits.push(t.label);
  if (p.width) bits.push(`${p.width}×${p.height}`);
  if (p.frames) bits.push(`${p.frames}f`);
  else if (p.duration) bits.push(`${p.duration.toFixed(1)}s`);
  if (t.seed !== undefined && t.seed !== null) bits.push(`seed ${t.seed}`);
  if (t.elapsed) bits.push(fmtSecs(t.elapsed));
  return bits.join(" · ");
}

function takePrompt(t) {
  return (t.params && t.params.common && t.params.common.prompt) || "";
}

function takeCaption(t) {
  const prompt = takePrompt(t);
  return [t.name, takeMeta(t), prompt ? `“${prompt.length > 200 ? `${prompt.slice(0, 200)}…` : prompt}”` : ""].filter(Boolean).join("  ·  ");
}

function visibleTakes() {
  return $("starFilter").checked ? S.takes.filter((t) => t.starred) : S.takes;
}

function opButton(text, title, fn, cls = "ghost") {
  return el("button", { class: cls, type: "button", text, title,
    onclick: async (e) => { e.stopPropagation(); closePopmenus(); try { await fn(); } catch (err) { toast(err.message, { kind: "error" }); } } });
}

function downloadLink(url, name) {
  return el("a", { class: "menu-link", role: "menuitem", href: url, download: name, text: "Download",
    onclick: (e) => { e.stopPropagation(); closePopmenus(); } });
}

function renderTakes() {
  const takes = visibleTakes();
  const starredCount = S.takes.filter((t) => t.starred).length;
  $("takeCount").textContent = S.takes.length ? String(S.takes.length) : "";
  $("starFilterCount").textContent = starredCount ? `(${starredCount})` : "";
  if (!takes.length) {
    $("takeList").replaceChildren(el("li", { class: "list-empty",
      text: $("starFilter").checked ? "No starred takes in this session." : "No takes yet in this session." }));
    renderCompareBar();
    return;
  }
  $("takeList").replaceChildren(...takes.map((t) => {
    const comparing = S.compare.includes(t.name);
    const previews = t.previews || [];
    const menu = popmenu("⋮", "More actions", [
      menuItem("First frame → inputs", "Extract the first frame into inputs", () => extractFrame(t, "first")),
      t.probe && t.probe.has_audio ? menuItem("Use audio", "Extract the audio track into inputs (audio → video)", () => extractAudio(t)) : null,
      previews.length ? menuItem(`Previews (${previews.length})`, "Scrub through the live previews saved during this render", async () => openScrubber(t)) : null,
      menuItem(comparing ? "Remove from compare" : "Add to compare", "Pick takes for the seed grid or A/B wipe", async () => toggleCompare(t.name)),
      downloadLink(t.url, t.name),
      menuItem("Delete…", "Delete this take, its settings and previews", () => deleteTake(t), "danger-item"),
    ]);
    return el("li", { class: [t.name === S.selectedTake ? "on" : "", comparing ? "comparing" : ""].join(" ").trim(),
      dataset: { name: t.name }, onclick: () => selectTake(t.name) },
      el("div", { class: "row" },
        el("div", { class: "thumb" }, el("img", { src: t.thumb, alt: "", loading: "lazy" })),
        el("div", { class: "info" },
          el("div", { class: "nm", text: t.name, title: takePrompt(t) || t.name }),
          el("div", { class: "meta", text: takeMeta(t) })),
        el("button", { class: `star${t.starred ? " on" : ""}`, type: "button", title: t.starred ? "Unstar" : "Star",
          "aria-pressed": String(!!t.starred), text: t.starred ? "★" : "☆",
          onclick: (e) => { e.stopPropagation(); setStar(t, !t.starred); } })),
      el("div", { class: "ops" },
        opButton("Chain →", "Use the last frame as the start image of Image → Video", () => chainTake(t)),
        t.params ? opButton("Reuse", "Restore the task, inputs and settings of this take", async () => { restore(t.params); toast(`Settings restored from ${t.name}.`, { kind: "ok", timeout: 3000 }); }) : null,
        opButton("Use video", "Copy this take into inputs (retake, extend, control)", () => useVideo(t.name, "outputs")),
        opButton("Last frame", "Extract the last frame into inputs", () => extractFrame(t, "last")),
        menu));
  }));
  renderCompareBar();
}

function selectTake(name, fromUser = true) {
  if (fromUser && S.runningId) S.followPreview = false;
  const take = S.takes.find((t) => t.name === name);
  S.selectedTake = take ? name : null;
  S.selectedTimeline = null;
  if (!take) return showInViewer(null);
  showInViewer(take, takeCaption(take), fromUser);
  renderRunningFromQueue();
}

async function setStar(take, starred) {
  try {
    await api("/api/takes/star", { session: S.session, name: take.name, starred });
    take.starred = starred;
    renderTakes();
  } catch (err) {
    toast(err.message, { kind: "error" });
  }
}

async function deleteTake(t) {
  if (!confirm(`Permanently delete this take, its settings and previews?\n${t.name}\n\nThis cannot be undone.`)) return;
  await api("/api/takes/delete", { session: S.session, name: t.name });
  S.compare = S.compare.filter((n) => n !== t.name);
  if (S.selectedTake === t.name) showInViewer(null);
  await loadTakes();
}

// ── continuing from a take ───────────────────────────────────────────────

async function chainTake(t) {
  const { name } = await api("/api/frame", { session: S.session, take: t.name, position: "last" });
  await loadInputs();
  S.taskId = "i2v";
  taskValues("i2v").image = name;
  renderTask();
  saveSettings();
  toast(`${name} is now the start image of Image → Video.`, { kind: "ok", timeout: 4000 });
}

async function extractFrame(t, position) {
  const { name } = await api("/api/frame", { session: S.session, take: t.name, position });
  await loadInputs();
  toast(`Added ${name} to inputs — click it to fill a slot.`, { kind: "ok", timeout: 4000 });
}

async function extractAudio(t) {
  const { name } = await api("/api/audio", { session: S.session, take: t.name });
  await loadInputs();
  toast(`Added ${name} to inputs.`, { kind: "ok", timeout: 4000 });
}

async function useVideo(name, kind) {
  const res = await api("/api/use-video", { session: S.session, kind, name });
  await loadInputs();
  toast(`Copied ${res.name} into inputs.`, { kind: "ok", timeout: 4000 });
}

// ── compare: seed grid and A/B wipe ──────────────────────────────────────

function toggleCompare(name) {
  S.compare = S.compare.includes(name) ? S.compare.filter((n) => n !== name) : [...S.compare, name].slice(-4);
  renderTakes();
}

function renderCompareBar() {
  const n = S.compare.length;
  $("compareBar").hidden = n === 0;
  $("compareCount").textContent = `${n} picked`;
  $("compareGrid").disabled = n < 2;
  $("compareWipe").disabled = n !== 2;
}

/** Keep several videos in lockstep with the first one. */
function syncVideos(videos) {
  const [lead, ...rest] = videos;
  let syncing = false;
  const follow = (fn) => { if (syncing) return; syncing = true; rest.forEach(fn); syncing = false; };
  lead.addEventListener("play", () => follow((v) => v.play().catch(() => {})));
  lead.addEventListener("pause", () => follow((v) => v.pause()));
  lead.addEventListener("seeked", () => follow((v) => { v.currentTime = lead.currentTime; }));
  lead.addEventListener("ratechange", () => follow((v) => { v.playbackRate = lead.playbackRate; }));
  lead.addEventListener("timeupdate", () => follow((v) => {
    if (Math.abs(v.currentTime - lead.currentTime) > 0.12) v.currentTime = lead.currentTime;
  }));
  lead.addEventListener("ended", () => follow((v) => v.pause()));
  return lead;
}

function openCompare(content, caption) {
  hidePreview();
  $("player").pause();
  $("player").classList.remove("on");
  $("viewerEmpty").hidden = true;
  const box = $("compare");
  box.replaceChildren(...content);
  box.hidden = false;
  $("viewer").classList.add("comparing");
  setCaption(caption);
}

function openCompareGrid(names = S.compare) {
  const takes = names.map((name) => S.takes.find((t) => t.name === name)).filter(Boolean);
  if (takes.length < 2) return;
  closeCompare();
  const videos = takes.map((t) => el("video", { src: t.url, muted: true, playsInline: true, loop: true, preload: "auto" }));
  const cells = takes.map((t, i) => el("figure", { class: "compare-cell" }, videos[i],
    el("figcaption", {},
      el("span", { class: "nm", text: t.seed !== undefined && t.seed !== null ? `seed ${t.seed}` : t.name, title: t.name }),
      el("button", {
        class: `ghost sm keep${t.starred ? " on" : ""}`, type: "button", text: t.starred ? "★ Kept" : "☆ Keep",
        onclick: async (event) => {
          const button = event.currentTarget;
          await setStar(t, !t.starred);
          button.textContent = t.starred ? "★ Kept" : "☆ Keep";
          button.classList.toggle("on", !!t.starred);
        },
      }),
      el("button", { class: "ghost sm", type: "button", text: "Open", onclick: () => selectTake(t.name) }))));
  const lead = syncVideos(videos);
  const controls = el("div", { class: "compare-controls" },
    el("button", { class: "ghost sm", type: "button", text: "Play / pause", onclick: () => (lead.paused ? lead.play().catch(() => {}) : lead.pause()) }),
    el("button", { class: "ghost sm", type: "button", text: "Restart", onclick: () => { lead.currentTime = 0; lead.play().catch(() => {}); } }),
    el("span", { class: "hint", text: "Muted and looped in sync · Keep stars a take" }),
    el("button", { class: "ghost sm", type: "button", text: "Close", onclick: exitCompare }));
  openCompare([el("div", { class: `compare-grid n${takes.length}` }, cells), controls], `Comparing ${takes.length} takes`);
  videos.forEach((v) => v.addEventListener("loadeddata", () => { if (videos.every((x) => x.readyState >= 2)) lead.play().catch(() => {}); }, { once: true }));
}

function openWipe(names = S.compare) {
  const takes = names.map((name) => S.takes.find((t) => t.name === name)).filter(Boolean);
  if (takes.length !== 2) return;
  closeCompare();
  const [a, b] = takes;
  const videoA = el("video", { src: a.url, playsInline: true, loop: true, preload: "auto" });
  const videoB = el("video", { src: b.url, playsInline: true, muted: true, loop: true, preload: "auto" });
  const divider = el("div", { class: "wipe-divider" });
  const stage = el("div", { class: "wipe-stage" }, videoA, videoB, divider,
    el("span", { class: "wipe-label left", text: `A · ${a.name}` }),
    el("span", { class: "wipe-label right", text: `B · ${b.name}` }));
  const setWipe = (pct) => {
    videoB.style.clipPath = `inset(0 0 0 ${pct}%)`;
    divider.style.left = `${pct}%`;
  };
  const slider = el("input", { type: "range", min: "0", max: "100", value: "50", "aria-label": "Wipe position",
    oninput: (e) => setWipe(Number(e.target.value)) });
  stage.addEventListener("pointermove", (e) => {
    if (!(e.buttons & 1)) return;
    const rect = stage.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
    slider.value = String(pct);
    setWipe(pct);
  });
  setWipe(50);
  const lead = syncVideos([videoA, videoB]);
  openCompare([stage, el("div", { class: "compare-controls" },
    el("button", { class: "ghost sm", type: "button", text: "Play / pause", onclick: () => (lead.paused ? lead.play().catch(() => {}) : lead.pause()) }),
    slider,
    el("button", { class: "ghost sm", type: "button", text: "Close", onclick: exitCompare }))],
  "A/B wipe · drag across the video or use the slider");
}

/** Track a multi-seed submit; the seed grid opens once every job has finished. */
function trackCompareBatch(jobs) {
  S.compareBatch = jobs.length > 1 ? { ids: new Set(jobs.map((j) => j.id)), outputs: [] } : null;
}

function noteBatchJob(job) {
  const batch = S.compareBatch;
  if (!batch || !batch.ids.has(job.id) || !["done", "failed", "cancelled"].includes(job.status)) return;
  batch.ids.delete(job.id);
  if (job.status === "done" && job.output) batch.outputs.push(job.output.split("/").pop());
  if (batch.ids.size) return;
  S.compareBatch = null;
  if (batch.outputs.length < 2 || job.session !== S.session) return;
  S.batchOpening = true;
  loadTakes().then(() => {
    S.compare = batch.outputs.filter((name) => S.takes.some((t) => t.name === name)).slice(0, 4);
    renderTakes();
    openCompareGrid(S.compare);
  }).finally(() => { S.batchOpening = false; });
}

// ── timeline list (sequences exported from helmstudio's timeline) ────────

async function loadTimeline() {
  S.timeline = await api(`/api/timeline?session=${encodeURIComponent(S.session)}`);
  renderTimelineList();
}

function timelineMeta(t) {
  const p = t.probe || {};
  const bits = [];
  if (p.width) bits.push(`${p.width}×${p.height}`);
  if (p.duration) bits.push(`${p.duration.toFixed(1)}s`);
  return bits.join(" · ") || "sequence";
}

function renderTimelineList() {
  $("timelineCount").textContent = S.timeline.length ? String(S.timeline.length) : "";
  if (!S.timeline.length) {
    $("timelineList").replaceChildren(el("li", { class: "list-empty", text: "No sequences yet. Create Timeline above makes one." }));
    return;
  }
  $("timelineList").replaceChildren(...S.timeline.map((t) => el("li", {
    class: t.name === S.selectedTimeline ? "on" : "", dataset: { name: t.name }, onclick: () => selectTimeline(t.name),
  },
    el("div", { class: "row" },
      el("div", { class: "thumb" }, el("img", { src: t.thumb, alt: "", loading: "lazy" })),
      el("div", { class: "info" },
        el("div", { class: "nm", text: t.name, title: t.name }),
        el("div", { class: "meta", text: timelineMeta(t) }))),
    el("div", { class: "ops" },
      opButton("Use video", "Use this sequence as an input (retake, extend, control)", () => useVideo(t.name, "timeline")),
      popmenu("⋮", "More actions", [
        downloadLink(t.url, t.name),
        menuItem("Delete…", "Delete this exported sequence", async () => {
          if (!confirm(`Delete this exported sequence?\n${t.name}\n\nThe sequence and its clips stay on helmstudio's timeline.`)) return;
          await api("/api/timeline/delete", { session: S.session, name: t.name });
          if (S.selectedTimeline === t.name) showInViewer(null);
          await loadTimeline();
        }, "danger-item"),
      ])))));
}

function selectTimeline(name) {
  S.followPreview = false;
  const item = S.timeline.find((t) => t.name === name);
  S.selectedTimeline = item ? name : null;
  S.selectedTake = null;
  if (!item) return showInViewer(null);
  showInViewer(item, [item.name, timelineMeta(item)].join("  ·  "), true);
  renderRunningFromQueue();
}
