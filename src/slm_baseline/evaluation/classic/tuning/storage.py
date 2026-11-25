"""Optuna study storage: journal file (default) or SQLAlchemy URL."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def study_storage(
    storage_url: str | None,
    *,
    journal_path: Path,
) -> str | Any:
    """Build Optuna ``storage`` for :func:`optuna.create_study`.

    Args:
        storage_url: If set, SQLAlchemy URL (e.g. Postgres on RDS).
        journal_path: Append-only journal file under ``runs/`` when ``storage_url`` is empty.

    Returns:
        Storage backend accepted by Optuna.
    """
    if storage_url and str(storage_url).strip():
        return str(storage_url).strip()
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(journal_path), lock_obj=None))


def clear_journal_storage(journal_path: Path) -> None:
    if journal_path.is_file():
        journal_path.unlink()