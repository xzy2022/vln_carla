from __future__ import annotations

from typing import Protocol

from usecases.dtos import EpisodeSummary


class RunEpisodeInputPort(Protocol):
    def run(self) -> EpisodeSummary:
        ...
