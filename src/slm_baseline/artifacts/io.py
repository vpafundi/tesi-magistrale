"""Small I/O helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as UTF-8 YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
