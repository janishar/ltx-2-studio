"""Virtual packs over official Lightricks LTX-2.5 weights (loader.official_pack).

The fast tests build tiny synthetic "official" files, so they need no weights.
The slow contract test runs against a real download when
``LTX25_OFFICIAL_DIR`` points at one.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.loader import official_pack as op


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Minimal safetensors writer for synthetic fixtures (float32 / uint8 only)."""
    dtypes = {np.dtype("float32"): "F32", np.dtype("uint8"): "U8"}
    header: dict[str, object] = {"__metadata__": metadata or {}}
    blobs, offset = [], 0
    for key, array in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[key] = {
            "dtype": dtypes[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    payload = json.dumps(header).encode()
    payload += b" " * (-len(payload) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for blob in blobs:
            f.write(blob)


def _rand(*shape: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


@pytest.fixture
def official_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ltx-2.5"
    dit_config = json.dumps({"transformer": {"ff_bias": False, "text_encoder_norm_type": "PER_TOKEN_RMS"}})
    _write_safetensors(
        root / "diffusion_models" / op.OFFICIAL_FILENAMES["transformer_distilled"],
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": _rand(128, 64),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": _rand(64, 128),
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": _rand(64, 64),
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.bias": _rand(64),
            "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight": _rand(32, 60),
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight": _rand(64, 16),
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight": _rand(
                8, 8
            ),
        },
        {"config": dit_config, "model_version": "2.5.0"},
    )
    _write_safetensors(
        root / "text_encoders" / op.OFFICIAL_FILENAMES["text_encoder"],
        {
            "model.layers.0.self_attn.q_proj.weight": _rand(64, 64),
            "model.layers.0.input_layernorm.weight": _rand(64),
            "model.embed_tokens.weight": _rand(128, 64),
            "text_embedding_projection.video_aggregate_embed.weight": _rand(64, 64),
            **{key: np.frombuffer(f"asset:{key}".encode(), dtype=np.uint8).copy() for key in op.TEXT_ENCODER_ASSETS},
        },
        {"gemma_config": json.dumps({"model_type": "gemma4_unified"})},
    )
    _write_safetensors(
        root / "vae" / op.OFFICIAL_FILENAMES["video_vae_conv"],
        {
            "encoder.conv_in.conv.weight": _rand(8, 3, 3, 3, 3),
            "decoder.conv_out.conv.weight": _rand(3, 8, 3, 3, 3),
            "per_channel_statistics.mean-of-means": _rand(8),
            "per_channel_statistics.std-of-means": _rand(8),
        },
    )
    _write_safetensors(
        root / "vae" / op.OFFICIAL_FILENAMES["video_vae_av"],
        {
            # The encoder half shares the file and must not reach the pack.
            "encoder.conv_in.conv.weight": _rand(8, 3, 3, 3, 3),
            "decoder.conv_in.weight": _rand(16, 8),
            "decoder.det_stages.0.0.attn.qkv.weight": _rand(48, 16),
            "decoder.diff_blocks.0.scale_shift_table": _rand(7, 16),
            "decoder.type_emb": _rand(8),
            "per_channel_statistics.mean-of-means": _rand(8),
            "per_channel_statistics.std-of-means": _rand(8),
        },
        {"config": json.dumps({"vae": {"decoder": {"_class_name": "NADiffusionDecoder"}}})},
    )
    _write_safetensors(
        root / "vae" / op.OFFICIAL_FILENAMES["audio_vae"],
        {
            "audio_vae.decoder.conv_in.conv.weight": _rand(8, 4, 3, 3),
            "audio_vae.per_channel_statistics.mean-of-means": _rand(8),
            "vocoder.vocoder.ups.0.weight": _rand(8, 4, 5),
            "vocoder.vocoder.conv_pre.weight": _rand(8, 4, 7),
            "vocoder.bwe_generator.conv_post.weight": _rand(1, 8, 7),
            "vocoder.mel_stft.stft_fn.forward_basis": _rand(16, 1, 32),
        },
    )
    # Off the upstream path on purpose: files are located by name.
    _write_safetensors(
        root / "latent_upscale_models" / op.OFFICIAL_FILENAMES["duration_head"],
        {"duration_head.mlp_hidden.weight": _rand(8, 8)},
    )
    _write_safetensors(
        root / "latent_upscale_models" / op.OFFICIAL_FILENAMES["spatial_upscaler"],
        {
            "upsampler.0.weight": _rand(16, 8, 3, 3),
            "res_blocks.0.conv1.weight": _rand(8, 8, 3, 3, 3),
            "initial_norm.weight": _rand(8),
        },
        {"config": json.dumps({"dims": 3})},
    )
    return root


@pytest.fixture
def pack(official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(op.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(op.QUANTIZE_ENV, "8")
    op._resolve_cached.cache_clear()
    return op.resolve_official_model_dir(official_dir)


def test_non_official_dir_is_unchanged(tmp_path: Path):
    (tmp_path / "transformer.safetensors").write_bytes(b"")
    assert op.resolve_official_model_dir(tmp_path) == tmp_path


def test_virtual_pack_sidecars(pack: Path):
    embedded = json.loads((pack / "embedded_config.json").read_text())
    assert embedded["model_version"] == "2.5.0"
    assert embedded["transformer"]["text_encoder_norm_type"] == "per_token_rms"
    assert json.loads((pack / "text_encoder_config.json").read_text())["model_type"] == "gemma4_unified"
    assert (pack / "tokenizer.json").read_bytes() == b"asset:tokenizer_json"
    assert json.loads((pack / "spatial_upscaler_x2_v1_0_config.json").read_text()) == {"config": {"dims": 3}}
    assert json.loads((pack / "quantize_config.json").read_text())["quantization"]["bits"] == 8
    for name in (
        "transformer.safetensors",
        "connector.safetensors",
        "vae_decoder_conv.safetensors",
        "vae_decoder_av.safetensors",
        "duration_head.safetensors",
    ):
        assert op.is_virtual_file(pack / name)


def test_transformer_keys_renamed_and_quantized(pack: Path):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    weights = load_split_safetensors(pack / "transformer.safetensors", prefix="transformer.")
    assert "transformer_blocks.0.ff.proj_in.weight" in weights
    assert "transformer_blocks.0.ff.proj_out.scales" in weights
    assert "transformer_blocks.0.attn1.to_out.weight" in weights
    assert "adaln_single.emb.timestep_embedder.linear1.weight" in weights
    # Last dim 60 is not a multiple of the group size: kept unquantized.
    assert weights["transformer_blocks.0.attn1.to_gate_logits.weight"].dtype == mx.float32
    assert not any(k.startswith("video_embeddings_connector") for k in weights)

    header_keys = set(op.read_safetensors_header(pack / "transformer.safetensors")[0])
    assert {f"transformer.{k}" for k in weights} == header_keys


def test_quantized_matches_mx_quantize(pack: Path, official_dir: Path):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    source = mx.load(str(official_dir / "diffusion_models" / op.OFFICIAL_FILENAMES["transformer_distilled"]))
    expected = mx.quantize(
        source["model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"], bits=8, group_size=64
    )
    weights = load_split_safetensors(pack / "transformer.safetensors", prefix="transformer.")
    for got, want in zip(
        (weights[f"transformer_blocks.0.ff.proj_in.{s}"] for s in ("weight", "scales", "biases")), expected, strict=True
    ):
        assert mx.array_equal(got, want)


def test_connector_keeps_sequential_names(pack: Path):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    weights = load_split_safetensors(pack / "connector.safetensors", prefix="connector.")
    assert list(weights) == ["video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"]


def test_prefix_filters_before_quantizing(pack: Path):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    projection = load_split_safetensors(
        pack / "text_encoder.safetensors", prefix="text_encoder.text_embedding_projection."
    )
    assert list(projection) == ["video_aggregate_embed.weight"]
    tower = load_split_safetensors(pack / "text_encoder.safetensors", prefix="text_encoder.model.")
    assert "layers.0.self_attn.q_proj.scales" in tower
    assert "embed_tokens.weight" in tower and "embed_tokens.scales" not in tower


def test_conv_transposition(pack: Path, official_dir: Path):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    decoder = load_split_safetensors(pack / "vae_decoder_conv.safetensors", prefix="vae_decoder_conv.")
    assert decoder["conv_out.conv.weight"].shape == (3, 3, 3, 3, 8)
    assert set(decoder) == {"conv_out.conv.weight", "per_channel_statistics.mean", "per_channel_statistics.std"}
    encoder = load_split_safetensors(pack / "vae_encoder_conv.safetensors", prefix="vae_encoder_conv.")
    assert "per_channel_statistics._mean_of_means" in encoder

    vocoder = load_split_safetensors(pack / "vocoder.safetensors", prefix="vocoder.")
    assert vocoder["ups.0.weight"].shape == (4, 5, 8)  # ConvTranspose1d (I, O, K) -> (O, K, I)
    assert vocoder["conv_pre.weight"].shape == (8, 7, 4)
    assert vocoder["mel_stft.stft_fn.forward_basis"].shape == (16, 32, 1)

    upscaler = load_split_safetensors(pack / "spatial_upscaler_x2_v1_0.safetensors")
    assert upscaler["spatial_upscaler_x2_v1_0.upsampler.0.weight"].shape == (16, 3, 3, 8)


def test_bf16_mode_writes_no_quantization(official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ltx_core_mlx.utils.weights import load_split_safetensors

    monkeypatch.setenv(op.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(op.QUANTIZE_ENV, "none")
    op._resolve_cached.cache_clear()
    pack = op.resolve_official_model_dir(official_dir)
    assert not (pack / "quantize_config.json").exists()
    weights = load_split_safetensors(pack / "transformer.safetensors", prefix="transformer.")
    assert not any(k.endswith(".scales") for k in weights)


def test_diffusion_decoder_is_the_decoder_half_unquantized(pack: Path, official_dir: Path):
    """``ltx-2.5-video-vae-bf16`` becomes ``vae_decoder_av.safetensors``: decoder half, as-is."""
    from ltx_core_mlx.utils.weights import load_split_safetensors

    weights = load_split_safetensors(pack / "vae_decoder_av.safetensors", prefix="vae_decoder_av.")
    assert set(weights) == {
        "conv_in.weight",
        "det_stages.0.0.attn.qkv.weight",
        "diff_blocks.0.scale_shift_table",
        "type_emb",
        "per_channel_statistics.mean",
        "per_channel_statistics.std",
    }
    # Linear-only: nothing is transposed on the way in, unlike the conv decoder.
    source = mx.load(str(official_dir / "vae" / op.OFFICIAL_FILENAMES["video_vae_av"]))
    assert mx.array_equal(weights["conv_in.weight"], source["decoder.conv_in.weight"])
    # Never quantized -- load_diffusion_decoder loads strictly into an unquantized module.
    header = op.read_safetensors_header(pack / "vae_decoder_av.safetensors")[0]
    assert not [key for key in header if key.endswith((".scales", ".biases"))]


def test_diffusion_decoder_config_reaches_the_placeholder(pack: Path):
    """``DiffusionDecoderConfig.from_safetensors_metadata`` reads the pack file, not the source."""
    from ltx_core_mlx.model.video_vae.diffusion_decoder import DiffusionDecoderConfig

    metadata = op.read_safetensors_header(pack / "vae_decoder_av.safetensors")[1]
    assert json.loads(metadata["config"])["vae"]["decoder"]["_class_name"] == "NADiffusionDecoder"
    # The synthetic config carries only the class name, so the real read raises on the
    # missing fields rather than silently falling back -- proof it reads this file.
    with pytest.raises(KeyError):
        DiffusionDecoderConfig.from_safetensors_metadata(pack / "vae_decoder_av.safetensors")


def test_the_av_vae_is_optional(official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A download without the diffusion decoder still resolves; the pack just omits it."""
    (official_dir / "vae" / op.OFFICIAL_FILENAMES["video_vae_av"]).unlink()
    monkeypatch.setenv(op.CACHE_ENV, str(tmp_path / "cache-no-av"))
    monkeypatch.setenv(op.QUANTIZE_ENV, "8")
    op._resolve_cached.cache_clear()
    pack = op.resolve_official_model_dir(official_dir)
    assert (pack / "vae_decoder_conv.safetensors").exists()
    assert not (pack / "vae_decoder_av.safetensors").exists()


_OFFICIAL_DIR = os.environ.get("LTX25_OFFICIAL_DIR")


@pytest.mark.slow
@pytest.mark.skipif(_OFFICIAL_DIR is None, reason="set LTX25_OFFICIAL_DIR to an official LTX-2.5 download")
def test_real_download_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Key counts match mlx-forge's EXPECTED_TENSOR_COUNTS; DiT keys match the model tree."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    monkeypatch.setenv(op.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(op.QUANTIZE_ENV, "8")
    op._resolve_cached.cache_clear()
    pack = op.resolve_official_model_dir(_OFFICIAL_DIR)

    expected = {
        "vae_encoder_conv": 86,
        "vae_decoder_conv": 86,
        "vae_encoder_av": 86,
        "vae_decoder_av": 312,
        "audio_vae": 102,
        "vocoder": 1227,
        "duration_head": 15,
        "spatial_upscaler_x2_v1_0": 72,
        "text_encoder": 681,
        "connector": 258,
    }
    for component, count in expected.items():
        keys = op.read_safetensors_header(pack / f"{component}.safetensors")[0]
        assert sum(not k.endswith((".scales", ".biases")) for k in keys) == count, component

    keys = set(op.read_safetensors_header(pack / "transformer.safetensors")[0])
    model = LTXModel(LTXModelConfig.from_checkpoint_dir(pack))
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"transformer.{p}.scales" in keys,
    )
    assert {f"transformer.{k}" for k, _ in tree_flatten(model.parameters())} == keys
