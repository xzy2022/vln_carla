from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeSummary:
    total_steps: int
    total_reward: float
