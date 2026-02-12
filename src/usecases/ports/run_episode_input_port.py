from __future__ import annotations

from typing import Protocol

from usecases.episode_types import EpisodeResult, EpisodeSpec


class RunEpisodeInputPort(Protocol):
    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        ...
