# ltx studio

A local web control surface for every `ltx-2-mlx` command — text, image, audio
and video to video, retake/extend, keyframes, IC-LoRA control, prompt tools and
training — from one browser tab, with nothing sent off your machine.

It is a Python server (`server.py`, no web framework, so it runs in the repo's
existing uv environment) plus a no-build-step vanilla JS front end (`static/`).
The layout follows [h3 studio](https://github.com/janishar/h3c-studio), and every
colour, font and radius is a helm-css token, so the page follows helmstudio's
light and dark themes and wears the studio's hue from `helmstudio.yaml`; the
declarative task catalog follows AuK Studio.

It keeps nothing of its own: [helmstudio](https://github.com/janishar/helmstudio)
keeps everything, through its runtime SDK — see
[Where things are kept](#where-things-are-kept).

## Running

ltx studio runs under helmstudio, which starts it from `helmstudio.yaml`, or on
its own under `helm dev`, helmstudio's CLI, which keeps everything in `.helm/`
beside the manifest. Started any other way it exits and says so. `web/run.sh`
starts it under helm dev, from a terminal or from VS Code; extra arguments go to
helm dev. It needs two things helmstudio publishes:

- **`helm`**, installed with helmstudio's installer, the first line below: it
  downloads the newest release for this Mac, checks it against the release's
  `SHA256SUMS`, and puts `helm` in `~/.local/bin`. Running it again updates
  `helm`; [Installing helm](https://github.com/janishar/helmstudio/blob/main/docs/releasing.md#installing-helm)
  covers a particular version and uninstalling. Kubernetes' CLI is also called
  `helm`; if that one comes first on `PATH`, set `HELM` to helmstudio's.
- **`helm-runtime-sdk`**, from PyPI: a dependency in `pyproject.toml`, pinned in
  `uv.lock`, so `uv sync` installs it into `.venv`.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/janishar/helmstudio/main/installer/install.sh)"   # helm, once
LTX_MODEL=~/models/LTX-2.5 bash web/run.sh    # port 8720, 127.0.0.1
```

The page needs nothing installed. It loads helm-css, the runtime's browser
client and helmstudio's components from whatever runs the studio, at
`/helm/sdk/v1` through the SDK's proxy in `server.py`: `helm`'s own copy under
helm dev, and helmstudio's inside helmstudio. Nothing comes from npm, a CDN or
a clone of helmstudio. [How a studio fits together](https://helmstudio.in/docs/concepts/how-a-studio-fits-together/)
explains the parts.

- `LTX_MODEL`, a local directory of the official Lightricks LTX-2.5 files, in
  any layout: each file `helmstudio.yaml` declares is linked to the file of the
  same name there, and helm dev uses the links. Without it, helm dev uses the
  weights linked before; it never downloads them, so with nothing linked the
  studio does not start.
- `HELM`, the `helm` to run: `helm` on `PATH` by default.
- `LTX_DEBUGPY`, a port: the server listens there for a Python debugger, with
  debugpy installed into `.venv` the first time. helm dev passes the studio a
  restricted environment, so the variable cannot reach the server itself;
  instead helm dev is given an environment whose `python` is `.venv`'s, run
  under debugpy.

What the script makes itself (the weight links, the debugger's environment and
its pid file) stays in `.cache/ltx-studio`, beside the `.helm` helm dev keeps,
and survives a restart. helm dev records where a weight is linked, so the links
always go there. `bash web/run.sh stop` stops the studio the script last started.
Starting it again stops the previous one first.

Open http://127.0.0.1:8720. The model can also be changed from the **Model**
button, which shows what the model can run (distilled / dev transformer, LTX-2.5
vs 2.3); `--gemma` sets the Gemma 3 repo used by LTX-2.3 packs and prompt
enhancement.

There is **no authentication** — keep it bound to `127.0.0.1` (see [Security](#security)).

`helmstudio.yaml` starts the server with these flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `{models.ltx}` | The `ltx` weight helmstudio resolved. Also a model directory, official LTX-2.5 files or Hugging Face repo id. |
| `--gemma` | `$LTX_GEMMA` | Gemma 3 repo for LTX-2.3 packs and prompt enhancement. |
| `--host` | `127.0.0.1` | Bind address. A warning is printed for anything but loopback. |
| `--port` | `{port}` | Bind port; helmstudio prefers 8720. |
| `--allow-host` | *(none)* | Extra `Host` names to accept, comma-separated. IP addresses and `localhost` are always accepted. |

### Run from VS Code

**ltx studio** starts the studio with `web/run.sh`, with `LTX_MODEL` set for
this machine in `.vscode/tasks.json`, and attaches the Python debugger to the
server; stopping the session stops the studio. The task **run: ltx studio**
starts it without the debugger. Static files are served uncached, so front-end
edits only need a browser refresh; restart for server changes. The other tasks
are **setup: uv sync** (with `--inexact`, so it keeps the debugpy `web/run.sh`
installs for the debugger), **test: fast suite** (default test task) and
**lint: ruff** (default build task).

| Configuration (`Cmd+Shift+D`, `F5`) | What it does |
| --- | --- |
| **ltx studio** | Runs the task **debug: ltx studio** (`web/run.sh` with `LTX_DEBUGPY=5678`) and attaches to the server once it is up; ending the session runs **stop: ltx studio**. Host and port are the manifest's, `127.0.0.1:8720`. |
| **ltx studio (step into ltx packages)** | The same, with `justMyCode` off. |
| **ltx-run: modes** | Prints which tasks the chosen model supports. |
| **ltx-2-mlx: generate (prompt)** | Runs one distilled generation in-process under the debugger (prompts for model and prompt), writing `outputs/vscode-generate.mp4`. Render jobs run in child processes, so this is the way to put breakpoints inside a render. |
| **pytest: fast suite** | `pytest -m "not slow"` under the debugger. |

## Using it

1. Pick a **task**. Tasks are grouped: Generate (text, image, first+last frame,
   multi-image anchors, prompt beats), Audio, Edit Video (retake, extend,
   keyframe interpolation), Control (IC-LoRA control, HDR, lip dub), Tools
   (prompt enhance, model info) and Training (slice, preprocess, train). Tasks
   the current model can't run explain why and stay disabled.
2. Drop or browse **inputs** (images, videos, audio). helmstudio keeps them as
   the session's inputs; click one to fill the next empty slot of the task, or
   pick it from a slot's menu.
3. Fill in the prompt, canvas (see [Canvas size](#canvas-size)), duration on
   the 8k+1 frame grid (or LTX-2.5 auto duration), seed, and task options. **Advanced** holds
   sampler knobs, LoRAs, quantize-on-load, low-RAM streaming, tiling and extra
   raw arguments. **Command** shows the exact `ltx-2-mlx` invocation.
   **History ▾** under the prompt brings back any prompt this session's takes used.
   The generate tasks also offer **Generated keyframes** for LTX-2.5 models
   (`--num-generated-keyframes`): extra keyframes at evenly spaced interior frames
   that the model generates with stage 1 to keep fast motion sharp. Each costs one
   latent frame of stage-1 tokens; 0 (the default) turns it off. The option is
   refused with an explanation on LTX-2.3 packs or when the clip has fewer than
   N + 2 frames.
4. **Render** (or ⌘/Ctrl+Enter) queues the job; **Queue 3 seeds** (⇧⌘/Ctrl+Enter)
   queues three random seeds and opens them side by side when they finish (see
   [Comparing takes](#comparing-takes)). Both sit in the render bar pinned to the
   bottom of the left pane, next to an estimate taken from this session's finished
   takes (`≈` when earlier takes had the same settings, `~` when scaled from takes
   of the same task at another size or length).
   One job runs at a time. The progress card shows an Encode → Load → Denoise →
   Decode → Save stepper, denoising step progress, elapsed time, the time left in
   the current denoising stage (from the pipeline's own `[estimate]` lines) and a
   **Stop** button; the tab title shows overall progress, and 🔔 in the top bar
   turns on a browser notification when a render finishes while the tab is in the
   background. A failed render pops up its error and, for known problems (out of
   memory, the macOS GPU watchdog, a missing DurationHead or dev transformer,
   Hugging Face access, ffmpeg), a hint about what to try.
5. Optionally tick **Live preview** before rendering to watch the video take
   shape — see [Live preview](#live-preview).
6. Every take appears under **Takes** on the right with its settings. From a take
   you can **Chain →** (last frame becomes the start image of Image → Video),
   **Reuse** its settings, pull the video itself (**Use video**, for retake, extend
   or control) or its **Last frame** into inputs. The **⋮** menu adds the first
   frame, the audio track (for audio → video), **Previews (N)**, **Add to compare**,
   **Download** and **Delete…**. ☆ stars a take and **★ starred only** filters the
   list. The **Timeline** tab lists the sequences helmstudio holds for this
   studio and the files exported from them, and **Gallery** in the top bar
   browses every take this studio made.

The terminal is helmstudio's `helm-terminal`, streaming the log of the
session's latest render from helmstudio, which keeps it; a render of the session
takes its place when it starts. Drag its top edge to resize it.
With more than eight inputs of mixed kinds, chips above the library filter it by
kind.

## Comparing takes

- **Seed grid** — after **Queue 3 seeds** finishes, the viewer shows the takes
  muted, looped and playing in sync, each with **☆ Keep** (stars the take) and
  **Open**. Any 2–4 takes picked with **Add to compare** open the same grid from
  **Grid** in the compare bar.
- **A/B wipe** — with exactly two takes picked, **A/B wipe** overlays them: drag
  across the video or use the slider to move the split.

Esc or **Close** returns to the selected take.

## Keyboard

| Key | Action |
| --- | --- |
| ⌘/Ctrl+Enter | Render |
| ⇧⌘/Ctrl+Enter | Queue 3 seeds |
| Space | Play / pause the viewer |
| ← / → | Step one frame back / forward |
| J / K | Next / previous take |
| Esc | Close dialogs, menus and the compare view |

Space, arrows and J/K are ignored while typing in a field.

## Canvas size

Pick an **aspect ratio** and drag **Megapixels** (0.1–2.1 MP); the studio
solves the closest legal width × height. LTX needs sizes on a pixel grid:
two-stage pipelines (distilled, two-stage, HQ, audio → video, keyframe,
IC-LoRA, lip dub) render stage 1 at half size, so their sizes step in
multiples of **64**; `generate` with the one-stage pipeline renders at full
size and steps in multiples of **32**. Switching pipeline re-fits the size to
the new grid.

- **Aspect ratio** — 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3, 21:9; **Match
  input** uses the task's selected image or video; **Custom** keeps the current
  size and locks its ratio for the slider.
- **Readout** — resolved size, actual megapixels and ratio (the grid can shift
  the ratio a few percent), latent size (width/32 × height/32) and the grid.
- **Presets and Width/Height** — preset chips and typed sizes still work;
  typed values snap to the grid when you leave the field, and the aspect and
  megapixel controls follow.

Above 720p (0.9 MP) memory grows quickly with duration; beyond 1080p consider
tiling in **Advanced**.

## Live preview

Off by default. Tick **Live preview** (above the Render button) to have the
pipeline decode a short animated WebP of the current latent while it denoises;
each one appears in the viewer as soon as it is written, with a badge showing
its step and stage (e.g. `Preview step 3/8 · stage 1`). Available for
generate, audio → video, retake, extend, keyframe, IC-LoRA, HDR and lip dub;
the option hides for other tasks.

| Setting | CLI flag | Effect |
| --- | --- | --- |
| **Every N steps** | `--stepwise-interval` | Preview every N denoising steps (1–100). The final step of each stage is always previewed. |
| **Clip length** | `--stepwise-frames` | Latent frames to decode: **Still frame** (1), **Short** (3 → 17 frames), **Default** (8 → 57 frames), **Long** (16 → 121 frames). Longer clips show more motion but decode slower. |
| **Position** | `--stepwise-frame` | Which part of the video the clip is centred on: **Middle**, **Start**, **End**, or **Custom** latent frame index (negative counts from the end). |

The hint under the settings shows the resulting cost. Each preview runs a VAE
decode, and the decoder stays loaded for the whole render, so expect a slower
render and higher peak memory. With `--low-ram` most of the memory saving is
lost. Image quality is fixed by the pipeline (WebP quality 90) and is not
adjustable.

While a job runs, clicking another take stops the viewer following the render;
**Show live preview** in the progress panel switches back. When the take is
done the viewer switches to the finished video, and the take's ⋮ menu gains
**Previews (N)**: a slider through every preview in order, with
**Back to video** to return. Previews are scratch, written to the stage
directory helmstudio gives the studio: helmstudio clears it when it stops the
studio, so a take keeps its **Previews (N)** until then. They are deleted at
once when the job fails, is stopped, or produced none.

## Timeline

**Create Timeline** opens helmstudio's timeline: a sequence helmstudio keeps,
edited in place — reorder, trim, dissolve, gain, undo — and exported by
helmstudio. The editor lists the sequences this studio may read and opens one;
**New sequence** and **+** pick takes from the gallery.

The **Timeline** tab lists two different things under one word, and each row
says which it is. A **sequence** is the edit itself, which helmstudio keeps:
there is no file here, so clicking one opens the editor on it — an edit is a
thing to open, the way a document is — and **Play** watches it as it stands,
clip by clip, in the viewer a take plays in. Sound is each clip's own, and the
cuts are as tight as swapping a source can be: a sequence that has to be heard
as it was cut, or to be frame-exact, is an export. An **export** is a file in
the gallery, rendered from a sequence, and behaves as a take does: it plays in
the viewer, **Use video** pulls it back into inputs (e.g. to extend it), and it
can be downloaded or deleted. Deleting an export leaves the sequence it came
from on helmstudio's timeline.

## Where things are kept

Everything goes through helmstudio's runtime SDK (`State` in `server.py`), and every
file the server writes goes where helmstudio says: no sessions directory, no
settings files, nothing in the browser's storage.

| What | Where helmstudio keeps it |
| --- | --- |
| Sessions | helmstudio sessions, listed, created, opened, duplicated and deleted through the SDK; the one opened last opens on start |
| A session's settings | the session's state document, saved as you edit |
| Inputs | pinned assets, listed in the session's state; a take made from an input names it as that take's input |
| Takes | written to helmstudio's stage directory, adopted into its asset store, and recorded in its gallery with the session, the settings, the prompt, the probe and the inputs |
| Renders | helmstudio jobs of their session, with their progress and their logs, which the terminal streams; helmstudio can cancel one |
| Live previews | helmstudio's stage directory, scratch |
| Sequences | helmstudio's timeline, read back for the **Timeline** tab; exports are gallery items |
| Folders from **Slice Clips** and **Preprocess Dataset** | the studio's data directory, where the next tool reads them by path: `outputs/<session id>/`; each file is adopted as an asset where it is (and so becomes read-only), and the folder is a record in the `folders` collection |
| The page's preferences: side panel, terminal height, notifications | a kv document, `ui/preferences` |
| The theme | helmstudio's own; the page follows it |

Switch sessions from the top bar; new, duplicate and delete live in the **⋯**
menu next to it. Following helmstudio:

- **Duplicate** copies a session's settings and inputs; its takes stay with the
  original session.
- Deleting a session removes it with its settings and inputs; its takes stay in
  helmstudio's gallery.
- Deleting a take removes it from the gallery; helmstudio reclaims its file once
  nothing uses it. An input is never reclaimed.

## Model and setup

The **Model** button shows a status dot: green when everything checks out, amber
for a Hugging Face repo id (not inspected until it downloads) or a missing
optional file, red when no model is set, the directory is missing, or
`ffmpeg`/`ffprobe` aren't on `PATH`. The dialog lists each check (transformer
variants, VAE decoder, spatial upscaler, ffmpeg, ffprobe) and what the model can
run; **Recheck** runs the checks again after you fix something.

## Security

ltx studio has no authentication, so it defends against the one thing a local
tool must: other websites and other machines driving it.

- It binds to `127.0.0.1` by default and warns for any other address. Anyone
  who can reach the port can run jobs.
- Requests whose `Host` header isn't an IP address, `localhost`, the `--host`
  value or an `--allow-host` name are refused, which blocks DNS rebinding.
- State-changing requests must come from the studio's own origin and carry a
  JSON content type (uploads: an `X-Filename` header), so a page you visit can't
  forge them.
- Only whitelisted `ltx-2-mlx` subcommands run, without a shell, and task inputs
  must be the session's inputs, which the server fetches from helmstudio.
- The page never holds helmstudio's token: its calls go through the runtime
  SDK's proxy at `/helm/`, which accepts only the studio's own origin.

## Adding a task

Edit `static/tasks.js` only: declare the subcommand, which shared blocks it uses
(prompt, canvas, duration, seed, low-RAM, tiling, quantize), its fields, an
availability check, and a `build()` that returns the argument list. The server
appends `--model`, `--gemma`, `--quantize-on-load` and `--output` itself and
only runs whitelisted subcommands, without a shell.

## Requirements and limits

- helmstudio, or its `helm` CLI for `helm dev`, installed with helmstudio's
  installer, and `helm-runtime-sdk` in `.venv`, which `uv sync` installs from
  PyPI.
- `ffmpeg`/`ffprobe` on `PATH` for media probing and frame/audio extraction.
  Thumbnails and timeline exports are helmstudio's.
- Jobs can be stopped, but a stopped job leaves no take.
- `info` on an official-weights directory lists no files: the CLI inspects the
  directory directly rather than the virtual pack.
- Uploads are stored as sent; format support is whatever ffmpeg and the
  pipelines accept.
