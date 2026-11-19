"""Backward-compatible re-exports; prefer ``slm_baseline.sequence``."""

from .sequence import core as _sequence_core
from .sequence.core import *  # noqa: F403

# Tests and callers may patch tokenizer loading on this module path.
_load_hf_tokenizer = _sequence_core._load_hf_tokenizer
