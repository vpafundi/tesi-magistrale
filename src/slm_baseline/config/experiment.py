"""Load experiment YAML, resolve paths, compute hashes, write manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..domain.labels import MICRO_PER_MACRO

_VAR = re.compile(r"\$\{([^}]+)\}")


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def expand_vars(obj: Any, env: dict[str, str] | None = None) -> Any:
    env = env if env is not None else dict(os.environ)

    def sub_str(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return env.get(key, "")

        return _VAR.sub(repl, s)

    if isinstance(obj, str):
        return sub_str(obj)
    if isinstance(obj, dict):
        return {k: expand_vars(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_vars(x, env) for x in obj]
    return obj


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = {**base}
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_merged(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extends = raw.pop("extends", None)
    if extends:
        parent = (path.parent / extends).resolve()
        if not parent.is_file():
            raise FileNotFoundError(f"extends not found: {parent}")
        base = load_yaml_merged(parent)
        raw = merge_dict(base, raw)
    return raw


def resolve_path(p: str | Path, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def canonical_config_hash(cfg: dict[str, Any]) -> str:
    def norm(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: norm(o[k]) for k in sorted(o)}
        if isinstance(o, list):
            return [norm(x) for x in o]
        if isinstance(o, Path):
            return str(o)
        return o

    blob = json.dumps(norm(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def splits_hash(df: pd.DataFrame) -> str:
    cols = [c for c in ("record_key", "split", "fold") if c in df.columns]
    sub = df[cols].sort_values(cols[0]).reset_index(drop=True)
    blob = sub.to_json(orient="split", date_format="iso")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class DatasetConfig:
    path: str
    text_column: str = "ocr_surrogates"
    sample: dict[str, Any] = field(default_factory=lambda: {"n": None, "frac": None, "seed": 42})


@dataclass
class ScopeConfig:
    macro_category: str | None = None


@dataclass
class SplitConfig:
    strategy: str = "holdout"
    seed: int = 42
    test_frac: float = 0.07
    val_frac: float = 0.05
    k: int = 5
    assignment_from: str | None = None


@dataclass
class PromptConfig:
    version: str = "minimal"


@dataclass
class SequenceConfig:
    max_seq_len_mode: str = "manual"
    max_seq_len: int = 15360
    tokenizers: list[str] = field(default_factory=list)
    headroom: float = 0.2
    multiple: int = 1024
    token_sample_rows: int | None = None


@dataclass
class SlmRunConfig:
    model: str
    modes: list[str]


@dataclass
class RunsConfig:
    classic: list[str] = field(default_factory=list)
    slm: list[dict[str, Any]] = field(default_factory=list)
    eval_split: str = "test"
    limit: int | None = None
    seed: int = 42
    gpu_mem: float = 0.85


@dataclass
class UnslothTrainingConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2.0e-4
    max_seq_len: int | None = None
    early_stopping_patience: int = 2
    # False: 1 riga train.jsonl = 1 esempio SFT. True: auto packing / padding-free di Unsloth.
    packing: bool = False


@dataclass
class TrainSamplingConfig:
    """Resampling train SFT (val invariato). Cap ``*_altro``; boost micro rare."""

    enabled: bool = False
    seed: int = 42
    max_altro: int | None = None
    min_non_altro: int | None = None


@dataclass
class TrainingConfig:
    enabled: bool = False
    base_model: str = ""
    output_subdir: str = "adapters"
    unsloth: UnslothTrainingConfig = field(default_factory=UnslothTrainingConfig)
    sampling: TrainSamplingConfig = field(default_factory=TrainSamplingConfig)


@dataclass
class SyncConfig:
    s3_prefix: str = ""


@dataclass
class PathsConfig:
    root: str = ""
    output_root: str = ""
    data_root: str = ""


@dataclass
class ExperimentConfig:
    experiment_id: str
    paths: PathsConfig
    dataset: DatasetConfig
    split: SplitConfig
    prompt: PromptConfig
    sequence: SequenceConfig
    runs: RunsConfig
    sync: SyncConfig
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        scope_raw = d.get("scope") or {}
        macro = scope_raw.get("macro_category")
        if macro is not None and macro not in MICRO_PER_MACRO:
            raise ValueError(f"scope.macro_category sconosciuta: {macro!r}")
        split_raw = dict(d.get("split") or {})
        training_raw = d.get("training") or {}
        unsloth_raw = training_raw.get("unsloth") or {}
        sampling_raw = training_raw.get("sampling") or {}
        return cls(
            experiment_id=d["experiment_id"],
            paths=PathsConfig(**d.get("paths", {})),
            dataset=DatasetConfig(**d.get("dataset", {})),
            split=SplitConfig(**split_raw),
            prompt=PromptConfig(**d.get("prompt", {})),
            sequence=SequenceConfig(**d.get("sequence", {})),
            runs=RunsConfig(**d.get("runs", {})),
            sync=SyncConfig(**d.get("sync", {})),
            scope=ScopeConfig(macro_category=macro),
            training=TrainingConfig(
                enabled=bool(training_raw.get("enabled", False)),
                base_model=str(training_raw.get("base_model", "")),
                output_subdir=str(training_raw.get("output_subdir", "adapters")),
                unsloth=UnslothTrainingConfig(
                    **{k: unsloth_raw[k] for k in UnslothTrainingConfig.__dataclass_fields__ if k in unsloth_raw}
                ),
                sampling=TrainSamplingConfig(
                    **{k: sampling_raw[k] for k in TrainSamplingConfig.__dataclass_fields__ if k in sampling_raw}
                ),
            ),
        )


def load_merged_experiment_dict(path: Path) -> dict[str, Any]:
    """Load YAML with extends, expand ${ENV}, fill default paths."""
    root = repo_root()
    env_defaults = {
        "TESI_REPO_ROOT": str(root),
        "TESI_OUTPUT_DIR": str(root / "outputs"),
        "TESI_DATA_DIR": str(root / "data"),
        "TESI_S3_PREFIX": "",
        **os.environ,
    }
    raw = load_yaml_merged(path)
    raw = expand_vars(raw, env_defaults)
    if not raw.get("paths", {}).get("root"):
        raw.setdefault("paths", {})["root"] = str(root)
    if not raw.get("paths", {}).get("output_root"):
        raw.setdefault("paths", {})["output_root"] = str(root / "outputs")
    if not raw.get("paths", {}).get("data_root"):
        raw.setdefault("paths", {})["data_root"] = str(root / "data")
    return raw


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_merged_experiment_dict(path))


def experiment_output_dir(cfg: ExperimentConfig) -> Path:
    root = Path(cfg.paths.output_root)
    return (root / "experiments" / cfg.experiment_id).resolve()


def experiment_output_dir_for_id(cfg: ExperimentConfig, experiment_id: str) -> Path:
    """Directory ``outputs/experiments/<experiment_id>`` nello stesso output root."""
    root = Path(cfg.paths.output_root)
    return (root / "experiments" / experiment_id).resolve()


def resolved_dataset_path(cfg: ExperimentConfig) -> Path:
    p = cfg.dataset.path
    path = Path(p)
    if path.is_absolute():
        return path
    dr = Path(cfg.paths.data_root)
    return (dr / path).resolve()


def build_manifest(
    cfg: ExperimentConfig,
    *,
    config_hash: str,
    dataset_path: Path,
    dataset_hash: str | None,
    splits_hash_value: str | None,
    prompt_hash: str | None,
    prompt_version: str,
    parent_splits_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(cfg.paths.root)
    try:
        import sklearn

        sklearn_v = sklearn.__version__
    except Exception:
        sklearn_v = None
    try:
        import pandas as pd

        pd_v = pd.__version__
    except Exception:
        pd_v = None
    comparable = parent_splits_hash or splits_hash_value
    return {
        "experiment_id": cfg.experiment_id,
        "config_hash": config_hash,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "splits_hash": splits_hash_value,
        "scope_macro": cfg.scope.macro_category,
        "split_assignment_from": cfg.split.assignment_from,
        "parent_splits_hash": parent_splits_hash,
        "comparable_group": comparable,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_hash,
        "git_commit": git_commit(root),
        "sklearn_version": sklearn_v,
        "pandas_version": pd_v,
    }
