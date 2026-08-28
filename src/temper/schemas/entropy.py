from __future__ import annotations

import math
from typing import List

from pydantic import field_validator, model_validator

from temper.schemas.base import MSONableModel


class EntropyProfilePoint(MSONableModel):
    """A single point of a QUESTS maximum-entropy entropy profile.

    Each point corresponds to one selection step. The step may add multiple
    frames; ``training_size`` is the selected-set size after that step. The
    "chunk" is the set of frames added since the previous point, so its size
    is the difference between consecutive ``training_size`` values.

    Attributes:
        training_size (int): Number of training frames selected at this point.
        cumulative_entropy (float): Cumulative QUESTS entropy of the training
            set at this point. May be infinite when QUESTS kernel values
            underflow, but must not be NaN. The QUESTS entropy normalizes by
            the set size, so it is not guaranteed to be non-decreasing as
            frames are added.
        information_gain (float): QUESTS information gain contributed by
            the chunk of frames added at this point, computed with the same
            ``delta_entropy`` objective used for selection. May be infinite
            when the corresponding entropy is infinite, but must not be NaN.
            The QUESTS ``delta_entropy`` is an unnormalized differential
            entropy (``-log(p)``) that is not bounded below by zero, so this
            value may legitimately be negative for redundant chunks.
    """

    training_size: int
    cumulative_entropy: float
    information_gain: float

    @field_validator("training_size")
    @classmethod
    def validate_training_size(cls, value: int) -> int:
        """Reject non-positive training sizes."""
        if value <= 0:
            raise ValueError(f"training_size must be positive, got {value}.")
        return value

    @field_validator("cumulative_entropy", "information_gain")
    @classmethod
    def validate_not_nan(cls, value: float) -> float:
        """Reject NaN while retaining signed infinite entropy sentinels."""
        if math.isnan(value):
            raise ValueError("Entropy values must not be NaN.")
        return value


class EntropyProfile(MSONableModel):
    """Ordered sequence of QUESTS entropy profile points.

    A trajectory may be persisted before evaluation with no profile
    (``entropy_profile=None``). When a profile is present it must be complete:
    the points must be ordered by strictly increasing ``training_size``.
    Profile points represent the owning trajectory's selection steps, so one
    point may represent a multi-frame chunk. Signed infinite values are
    retained as numerical saturation sentinels; NaN values are rejected.

    Validation is deliberately restricted to what the QUESTS objective
    guarantees (verified against ``quests==2026.2.22``):

    - Cumulative entropy normalizes by the set size and may increase or
      decrease as frames are added; no monotonicity or nonnegativity check is
      applied.
    - ``information_gain`` is a differential entropy (``-log(p)`` with an
      unnormalized kernel sum ``p``) and may legitimately be negative or
      positive infinite; only NaN is rejected.

    Attributes:
        points (list[EntropyProfilePoint]): Ordered entropy profile points.
    """

    points: List[EntropyProfilePoint]

    @model_validator(mode="after")
    def validate_profile(self) -> "EntropyProfile":
        """Validate ordering and reject present-but-empty profiles."""
        if not self.points:
            raise ValueError("EntropyProfile.points must not be empty.")
        previous_size = 0
        for i, point in enumerate(self.points):
            if point.training_size <= previous_size:
                raise ValueError(
                    "EntropyProfile points must have strictly increasing "
                    f"training_size; point {i} has training_size "
                    f"{point.training_size} after {previous_size}."
                )
            previous_size = point.training_size
        return self
