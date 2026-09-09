"""Temporal score aggregation operators."""

from .pooling import (
    apply_pool,
    attention_pool,
    generalized_mean,
    mean_pool,
    min_pool,
    power_mean,
    soft_min,
    soft_min_weighted,
    source_auto_pool,
    source_faithful_auto_pool,
    source_faithful_power_pool,
    source_power_pool,
)

__all__ = [
    "apply_pool",
    "attention_pool",
    "generalized_mean",
    "mean_pool",
    "min_pool",
    "power_mean",
    "soft_min",
    "soft_min_weighted",
    "source_auto_pool",
    "source_faithful_auto_pool",
    "source_faithful_power_pool",
    "source_power_pool",
]

