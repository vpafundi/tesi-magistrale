"""Eccezioni di dominio per la pipeline baseline; la CLI le mappa a exit code."""


class PipelineError(Exception):
    """Errore atteso nel flusso prepare / evaluate / sync."""


class InvalidMacroScopeError(PipelineError):
    """``scope.macro_category`` non presente in :data:`~slm_baseline.domain.labels.MICRO_PER_MACRO`."""


class EmptyScopeMacroError(PipelineError):
    """Dopo il filtro macro non restano segmenti nel dataset."""


class MasterSplitsNotFoundError(PipelineError):
    """``split.assignment_from`` referenzia un esperimento senza ``splits.parquet``."""


class DatasetNotFoundError(PipelineError):
    """Il parquet configurato in ``dataset.path`` non esiste sul filesystem."""

    def __init__(self, message: str) -> None:
        """Args: message: Dettaglio con percorso risolto."""
        super().__init__(message)


class ImmutableExperimentError(PipelineError):
    """Directory esperimento già presente con ``config_hash`` diverso."""

    def __init__(self, prev: str, new: str) -> None:
        """Args:
            prev: Hash nel ``manifest.json`` esistente.
            new: Hash della config corrente.
        """
        super().__init__(f"config_hash conflict: existing {prev} vs new {new}")
        self.prev_hash = prev
        self.new_hash = new


class UnknownSplitStrategyError(PipelineError):
    """Valore di ``split.strategy`` nel YAML diverso da ``holdout`` o ``kfold``."""


class PrepareArtifactsMissingError(PipelineError):
    """Valutazione richiesta prima di ``prepare`` (manca splits o run_config)."""


class UnknownClassicMethodError(PipelineError):
    """Nome in ``runs.classic`` non registrato in :class:`~slm_baseline.evaluation.classic.registry.ClassicMethodRegistry`."""


class TrainRunnerError(PipelineError):
    """Sottoprocesso ``run_unsloth`` terminato con codice non zero."""

    def __init__(self, message: str, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class SlmRunnerError(PipelineError):
    """Sottoprocesso ``run_baseline`` terminato con codice non zero."""

    def __init__(self, message: str, returncode: int | None = None) -> None:
        """Args:
            message: Comando e contesto.
            returncode: Exit code del subprocess, se disponibile.
        """
        super().__init__(message)
        self.returncode = returncode


class SyncConfigError(PipelineError):
    """``sync.s3_prefix`` assente, s3fs non installato o cartella output mancante."""
