"""Shared test fixtures and helpers."""

import os
from pathlib import Path

# MLX >= 0.32 enables TF32 for fp32 GPU matmuls by default, which costs ~8e-4 relative
# accuracy (a 10-bit mantissa) and is well outside the fp32 tolerances the numerics unit
# tests compare against their numpy references with. Production is unaffected -- every
# model runs in bf16 -- so restore true fp32 for the test session only. Must be set
# before mlx is first imported, which is why it lives at the top of conftest.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

# Weight-gated tests skip unless these point at local MLX packs (mlx-forge layout).
Q8_MODEL_ENV = "LTX_TEST_MODEL_DIR"  # LTX-2.3 int8 pack
LTX25_PACK_ENV = "LTX_TEST_LTX25_PACK_DIR"  # LTX-2.5 int8 pack

# A pack without this file cannot serve the weight-gated tests.
_REQUIRED_WEIGHT = "transformer-distilled.safetensors"


def find_q8_model_dir() -> Path | None:
    """The LTX-2.3 q8 pack named by ``$LTX_TEST_MODEL_DIR``, or None if unset or incomplete.

    A pack without the distilled transformer is treated as absent, so a
    partial download skips the weight-gated tests instead of failing them.
    """
    value = os.environ.get(Q8_MODEL_ENV)
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if (candidate / _REQUIRED_WEIGHT).exists() else None


MODEL_DIR = find_q8_model_dir()


def _local_pack(env: str) -> Path | None:
    """Local converted pack named by ``env``, used by the LTX-2.5 contract tests."""
    value = os.environ.get(env)
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if (candidate / "embedded_config.json").exists() else None


LTX25_Q8_DIR = _local_pack(LTX25_PACK_ENV)
