"""Stage training LoRA (delega a :class:`~slm_baseline.training.runner.TrainLoRAPipeline`)."""

from __future__ import annotations

from pathlib import Path

from ...training.runner import TrainLoRAPipeline
from ..context import ExperimentContext


def run_train_stage(ctx: ExperimentContext) -> Path | None:
    return TrainLoRAPipeline().run(ctx)