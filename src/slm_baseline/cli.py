"""Command-line interface for the thesis experiment workflow.

Exposes ``prepare``, ``classic``, ``train``, ``slm``, ``run``, ``compare``,
``compare-family``, and ``sync``; each command delegates to
:class:`BaselineExperimentOrchestrator`. This is the recommended entry point for
running an end-to-end experiment from the shell without notebooks or ad hoc scripts.

Usage examples::

    python -m slm_baseline.cli prepare --config configs/experiment.yaml
    python -m slm_baseline.cli run --config configs/experiment.yaml --rerun
    python -m slm_baseline.cli compare --output-root outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config.experiment import repo_root
from .logging.core import setup_pipeline_logger
from .pipeline.exceptions import PipelineError
from .pipeline.orchestrator import BaselineExperimentOrchestrator


def _output_root_from_args(args: argparse.Namespace) -> Path:
    """Return the outputs directory from ``--output-root`` or ``TESI_OUTPUT_DIR``."""
    if getattr(args, "output_root", None):
        return Path(args.output_root).resolve()
    return Path(os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))).resolve()


def _apply_output_root_env(path: Path | None) -> None:
    """Set ``TESI_OUTPUT_DIR`` to the output root selected for this CLI invocation."""
    if path is not None:
        os.environ["TESI_OUTPUT_DIR"] = str(path.resolve())


def _orchestrator(args: argparse.Namespace) -> BaselineExperimentOrchestrator:
    """Build an orchestrator, reusing the logger attached to ``args`` in ``main``."""
    logger = getattr(args, "_pipeline_logger", None)
    return BaselineExperimentOrchestrator(logger=logger)


def cmd_prepare(ns: argparse.Namespace) -> None:
    """Create the experiment directory, splits, and manifests from a config file."""
    _orchestrator(ns).prepare(Path(ns.config), force=ns.force)


def cmd_classic(ns: argparse.Namespace) -> None:
    """Run CPU baselines listed under ``runs.classic`` in the experiment config."""
    _orchestrator(ns).evaluate_classic(Path(ns.config), rerun=ns.rerun)


def cmd_train(ns: argparse.Namespace) -> None:
    """Start LoRA fine-tuning (Unsloth) when ``training.enabled`` is set."""
    _orchestrator(ns).train(Path(ns.config), rerun=ns.rerun, dry_run=ns.dry_run)


def cmd_slm(ns: argparse.Namespace) -> None:
    """Evaluate SLMs with vLLM (one subprocess per model/mode/fold combination)."""
    _orchestrator(ns).evaluate_slm(Path(ns.config), rerun=ns.rerun, dry_run=ns.dry_run)


def cmd_run(ns: argparse.Namespace) -> None:
    """Run prepare, classic baselines, and SLM evaluation in sequence for one experiment."""
    _orchestrator(ns).run_all(Path(ns.config), force=ns.force, rerun=ns.rerun)


def cmd_compare(ns: argparse.Namespace) -> None:
    """Aggregate metrics and artifacts from all experiment folders under the output root."""
    _orchestrator(ns).compare(_output_root_from_args(ns))


def cmd_compare_family(ns: argparse.Namespace) -> None:
    """Merge predictions across the four macro experiments and compute family-level metrics."""
    _orchestrator(ns).compare_family(
        ns.slug,
        family_prefix=ns.family_prefix,
        output_root=_output_root_from_args(ns),
    )


def cmd_sync(ns: argparse.Namespace) -> None:
    """Upload one or all experiment folders to S3 via ``s3fs``."""
    cfg = Path(ns.config).resolve() if ns.config else None
    _orchestrator(ns).sync(cfg, all_experiments=ns.all, output_root=_output_root_from_args(ns))


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging and output root, then dispatch the subcommand.

    :class:`~slm_baseline.pipeline.exceptions.PipelineError` instances are logged and
    turned into exit code 1 with a readable message.
    """
    p = argparse.ArgumentParser(prog="python -m slm_baseline.cli")
    p.add_argument(
        "--output-root",
        default=None,
        help="Override cartella outputs (imposta TESI_OUTPUT_DIR per questa invocazione)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Crea cartella esperimento, split, manifest, run_config")
    p_prep.add_argument("--config", required=True)
    p_prep.add_argument("--force", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_tr = sub.add_parser("train", help="Fine-tuning LoRA (Unsloth) per esperimento macro-scoped")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument("--rerun", action="store_true")
    p_tr.add_argument("--dry-run", action="store_true")
    p_tr.set_defaults(func=cmd_train)

    p_cl = sub.add_parser("classic", help="Esegue baseline classiche nella cartella esperimento")
    p_cl.add_argument("--config", required=True)
    p_cl.add_argument("--rerun", action="store_true")
    p_cl.set_defaults(func=cmd_classic)

    p_slm = sub.add_parser("slm", help="Lancia run_baseline per ogni modello (sottoprocesso)")
    p_slm.add_argument("--config", required=True)
    p_slm.add_argument("--rerun", action="store_true")
    p_slm.add_argument("--dry-run", action="store_true")
    p_slm.set_defaults(func=cmd_slm)

    p_run = sub.add_parser("run", help="prepare + classic + slm")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--rerun", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Aggrega metriche da tutti gli esperimenti")
    p_cmp.set_defaults(func=cmd_compare)

    p_cf = sub.add_parser("compare-family", help="Unisce predizioni 4 macro e metriche globali")
    p_cf.add_argument("--slug", required=True, help="Slug run (es. gemma-3-4b-it__constrained__L15360)")
    p_cf.add_argument("--family-prefix", default="holdout_s42")
    p_cf.set_defaults(func=cmd_compare_family)

    p_syn = sub.add_parser("sync", help="Copia cartella esperimento su S3 (s3fs)")
    p_syn.add_argument("--config", default=None)
    p_syn.add_argument("--all", action="store_true")
    p_syn.set_defaults(func=cmd_sync)

    args = p.parse_args(argv)
    oroot = Path(args.output_root).resolve() if args.output_root else None
    _apply_output_root_env(oroot)

    pipeline_logger = setup_pipeline_logger()
    args._pipeline_logger = pipeline_logger
    pipeline_logger.info("cli: command=%s", args.command)

    try:
        args.func(args)
        pipeline_logger.info("cli: command=%s completed", args.command)
    except PipelineError as e:
        pipeline_logger.error("cli: %s", e)
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()