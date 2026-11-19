"""Experiment pipelines and orchestration."""

from .context import EvalFoldContext, ExperimentContext, PrepareStats

__all__ = [
    "BaselineExperimentOrchestrator",
    "EvalFoldContext",
    "ExperimentContext",
    "PrepareStats",
]


def __getattr__(name: str):
    if name == "BaselineExperimentOrchestrator":
        from .orchestrator import BaselineExperimentOrchestrator

        return BaselineExperimentOrchestrator
    raise AttributeError(name)
