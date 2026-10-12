from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def validate_vector(
    vector: object,
    dimension: int,
    *,
    label: str,
) -> list[float]:
    """Return a normalized vector after strict finite-real validation."""

    message = f"{label} must contain exactly {dimension} finite real values"
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise ValueError(message)
    if len(vector) != dimension:
        raise ValueError(f"embedding dimension mismatch: {message}")

    validated: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(message)
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(message)
        validated.append(numeric)
    return validated
