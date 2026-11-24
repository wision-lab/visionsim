"""Shared helpers for the dToF emulation tests.

Monte-Carlo kernels are validated two ways: against closed-form properties of the
sampling process (single-hit per cycle, dead-time spacing) and statistically
against the single-pixel analytical forward models, which is the cross-check the two
model families exist for.
"""

from __future__ import annotations

import torch

N_TBINS = 64


def make_generator(seed: int = 0) -> torch.Generator:
    """Create a seeded CPU generator for deterministic sampling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
