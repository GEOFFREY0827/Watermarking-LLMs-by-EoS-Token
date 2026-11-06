"""Everything secret is derived from one owner key, so verification is reproducible from the key alone."""
from __future__ import annotations

import hashlib
import hmac

import numpy as np


def derive_seed(key: str, purpose: str) -> int:
    digest = hmac.new(key.encode(), purpose.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def key_rng(key: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(key, purpose))


def commitment(key: str) -> str:
    """Publish this before releasing the model; reveal the key only when claiming ownership."""
    return hashlib.sha256(("eoswm-commit:" + key).encode()).hexdigest()


def balanced_bits(key: str, n: int) -> list[int]:
    """Exactly n/2 ones (n even), in key-determined order."""
    bits = np.array([1] * (n // 2) + [0] * (n - n // 2))
    key_rng(key, "bits").shuffle(bits)
    return bits.tolist()
