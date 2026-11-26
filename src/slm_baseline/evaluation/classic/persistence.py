"""Persist fitted classic models (TF-IDF + LogReg) under ``runs/``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn


def model_joblib_path(artifacts_dir: Path, slug: str) -> Path:
    return artifacts_dir / f"model__{slug}.joblib"


def model_meta_path(artifacts_dir: Path, slug: str) -> Path:
    return artifacts_dir / f"model__{slug}.json"


def classic_model_ready(artifacts_dir: Path, slug: str) -> bool:
    return model_joblib_path(artifacts_dir, slug).is_file() and model_meta_path(artifacts_dir, slug).is_file()


def clear_classic_model(artifacts_dir: Path, slug: str) -> None:
    model_joblib_path(artifacts_dir, slug).unlink(missing_ok=True)
    model_meta_path(artifacts_dir, slug).unlink(missing_ok=True)
    for p in (
        artifacts_dir / f"model_card__{slug}.json",
        artifacts_dir / f"model_card__{slug}.md",
    ):
        p.unlink(missing_ok=True)


def save_tfidf_model(
    artifacts_dir: Path,
    slug: str,
    model: Any,
    *,
    method: str,
    macro_scope: str | None,
    seed: int,
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = model_joblib_path(artifacts_dir, slug)
    meta_path = model_meta_path(artifacts_dir, slug)
    joblib.dump(model, joblib_path)
    meta = {
        "method": method,
        "slug": slug,
        "macro_scope": macro_scope,
        "seed": seed,
        "params": params,
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return joblib_path, meta_path


def load_tfidf_model(artifacts_dir: Path, slug: str) -> tuple[Any, dict[str, Any]]:
    joblib_path = model_joblib_path(artifacts_dir, slug)
    meta_path = model_meta_path(artifacts_dir, slug)
    if not joblib_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Modello classic mancante per slug={slug!r} in {artifacts_dir}")
    model = joblib.load(joblib_path)
    if not hasattr(model, "vectorizer") or not hasattr(model, "clf"):
        raise TypeError(f"Artefatto {joblib_path} non è un TfidfLogRegModel compatibile")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def lora_adapter_ready(adapter_dir: Path) -> bool:
    """True se la directory contiene un adapter PEFT salvato da Unsloth."""
    if not adapter_dir.is_dir():
        return False
    return (adapter_dir / "adapter_config.json").is_file() and (
        (adapter_dir / "adapter_model.safetensors").is_file()
        or (adapter_dir / "adapter_model.bin").is_file()
    )