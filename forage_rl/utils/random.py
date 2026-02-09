"""Randomness utilities shared across experiments."""

from typing import Optional

import numpy as np


def derive_seed(base_seed: Optional[int], *stream_components: int) -> Optional[int]:
    """Derive a deterministic 32-bit seed for a sub-stream.

    The same inputs always produce the same output. Different component tuples
    produce independent derived seeds suitable for environment/agent/run splits.
    """
    if base_seed is None:
        return None

    seed_sequence = np.random.SeedSequence(
        [int(base_seed), *[int(component) for component in stream_components]]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

