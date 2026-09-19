# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x.y`), breaking changes bump `y` and additive
changes bump `z`. See [`docs/PIPELINE_MATURITY.md`](docs/PIPELINE_MATURITY.md)
for per-pipeline stability guarantees.

## [Unreleased]

### Features

- **Official LTX-2.5 weights, converted on load.** `--model` accepts a directory
  of the official Lightricks LTX-2.5 files; weights are converted to the MLX
  layout in memory and quantized with `--quantize-on-load {8,4,none}`. Only
  configs and tokenizer assets are cached, in `<repo>/.cache/`.
- **Mode launcher** (`scripts/ltx_run.py`) for every generation path, with a
  model capability check.
- **ltx studio** (`web/`): stdlib-only local web UI for every command, with
  sessions, input library, job queue, streaming log, take actions (reuse
  settings, chain, frames, video/audio reuse) and a timeline editor that
  combines clips across sessions.
- **Optional live preview** in ltx studio: stepwise previews streamed into the
  viewer while denoising, configurable interval, clip length and position, with
  a per-take preview scrubber.
- VS Code launch, task and settings configurations.
- **Diffusion video decoder** (`generate --video-decoder diffusion`, LTX-2.5),
  synced from upstream ltx-2-mlx 0.15.6: an opt-in alternative to the conv VAE
  decoder that finishes the latent→pixel step with one diffusion evaluation,
  sharper on fine detail and several times slower. `conv` stays the default and
  every existing decode path is byte-identical. Single tile in v1, refused above
  `LTX2_DIFFVAE_MAX_TOKENS` (512×768×49) before any generation starts.
  - Works on the **official Lightricks LTX-2.5 files**, not just mlx-forge packs:
    the optional `ltx-2.5-video-vae-bf16.safetensors` now becomes `vae_encoder_av`
    + `vae_decoder_av` in the virtual pack (`VIRTUAL_PACK_FORMAT` 2, so existing
    virtual packs rebuild — header-only, seconds). A download without that file
    still resolves and simply has no diffusion decoder. Note that a *truncated*
    copy of it now fails every command, as for any other official file.
  - **Not compatible with live previews**: previews decode one window per step,
    which this decoder cannot do cheaply. `generate`, ltx studio and the Python
    API each refuse the combination up front rather than silently using conv.
  - ltx studio exposes it as **Video decoder** on the generate tasks, refused with
    an explanation on models without the weights and when live preview is on.
  - **Tiled decode** (synced from upstream 0.15.6+, `effd115`): the decode is tiled
    automatically to fit `LTX2_VAE_DECODE_BUDGET_GB` (default: half of unified
    memory), so there is no longer a resolution ceiling —
    `LTX2_DIFFVAE_MAX_TOKENS` now applies only to a forced one-tile decode.
    `--diffvae-tile FRAMES HEIGHT WIDTH` overrides the tile size.
- **Generated keyframe slots** (`generate --num-generated-keyframes N`, LTX-2.5),
  synced from upstream ltx-2-mlx 0.15.5: N evenly spaced single-pixel-frame slots
  denoised with stage 1, on all four generate modes; refused before any Gemma load
  on packs without `use_keyframes_abs_pos_embedding`. ltx studio exposes it as
  **Generated keyframes** on the generate tasks.

### Fixed

- Diffusion decoder tile sizing on **official weights**: the decode budget charged
  the placeholder file's size (~32 KB) instead of the 0.78 GB of weights behind it,
  leaving the tile sizer that much headroom it did not have.
- `virtual_source_info` / `is_virtual_file` raised `MemoryError` on a file that is
  not safetensors (eight junk bytes decode to a header length near `2**63`, and the
  read was unbounded) instead of answering "not virtual". The header read is now
  bounded by the file size.
- Test suite: MLX 0.32 turns on TF32 for fp32 GPU matmuls by default (~8e-4
  relative error), which broke the fp32 numerics tests against their numpy
  references. `tests/conftest.py` restores true fp32 for the test session;
  production is unaffected, since every model runs in bf16.

### Changed

- LTX-2.5 renders now apply the learned keyframe absolute-position embedding to
  the first latent frame (synced from upstream 0.15.5). It was loaded but never
  applied, so 2.5 output was slightly off the reference on frame 0; 2.5 outputs
  shift, 2.3 output is unchanged. Also applies to the official LTX-2.5 files
  converted on load.

- The CLI no longer defaults to a hosted model: `--model` falls back to
  `$LTX_MODEL` and exits with an error when neither is set.
- Weight-gated tests read local pack paths from `LTX_TEST_MODEL_DIR` (LTX-2.3
  int8) and `LTX_TEST_LTX25_PACK_DIR` (LTX-2.5 int8) instead of a fixed
  Hugging Face cache location.
- Releases are cut manually with `scripts/bump_version.py`,
  `scripts/validate_versions.py` and `scripts/generate_changelog.py`; the
  automated release workflows were removed.

## 0.15.4 and earlier

This project is a standalone fork. Changes up to and including `0.15.4` are
recorded in the git history (`git log v0.15.4`, or `git log 461df21` if the
tag is not present locally).
