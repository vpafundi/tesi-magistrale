"""Protocolli di typing per stage e strategie."""

from __future__ import annotations

from typing import Protocol

from .context import ExperimentContext


class PipelineStage(Protocol):
    """Contratto per un macrostep della pipeline (prepare o evaluate).

    Ogni implementazione espone un ``name`` stabile e muta ``ctx`` in ``run``.
    """

    @property
    def name(self) -> str:
        """Identificatore breve dello stage (logging e debug)."""

    def run(self, ctx: ExperimentContext) -> None:
        """Esegue lo stage aggiornando ``ctx`` e/o artefatti su disco.

        Args:
            ctx: Contesto esperimento; precondizioni dipendono dallo stage.

        Returns:
            None. Effetti collaterali su ``ctx`` e file sotto ``ctx.paths``.
        """
