"""Shared ordered split fingerprint for cross-model audit trails."""

import hashlib

import numpy as np


def split_fingerprint(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> str:
    """Hash ordered split indices using the versioned split-fingerprint contract."""

    digest = hashlib.sha256()
    digest.update(b"css700-split-v1\0")
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        ordered = np.ascontiguousarray(values, dtype="<i8")
        digest.update(name.encode() + b"\0")
        digest.update(len(ordered).to_bytes(8, "little"))
        digest.update(ordered.tobytes())
    return digest.hexdigest()
