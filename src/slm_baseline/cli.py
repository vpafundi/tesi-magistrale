"""Experiment workflow: argparse facade over :class:`BaselineExperimentOrchestrator`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config.experiment import repo_root
from .logging.core import setup_pipeline_logger
from .pipeline.exceptions import PipelineError
from .pipeline.orchestrator import BaselineExperimentOrchestrator


def _output_root_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "output_root", None):
        return Path(args.output_root).resolve()
    return Path(os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))).resolve()


def _apply_output_root_env(path: Path | None) -> None:
    if path is not None:
        os.environ["TESI_OUTPUT_DIR"] = str(path.resolve())


def _orchestrator(args: argparse.Namespace) -> BaselineExperimentOrchestrator:
    logger = getattr(args, "_pipeline_logger", None)
    return BaselineExperimentOrchestrator(logger=logger)


def cmd_prepare(ns: argparse.Namespace) -> None:
    _orchestrator(ns).prepare(Path(ns.config), force=ns.force)


def cmd_classic(ns: argparse.Namespace) -> None:
    _orchestrator(ns).evaluate_classic(Path(ns.config), rerun=ns.rerun)


def cmd_slm(ns: argparse.Namespace) -> None:
    _orchestrator(ns).evaluate_slm(Path(ns.config), rerun=ns.rerun, dry_run=ns.dry_run)


def cmd_run(ns: argparse.Namespace) -> None:
    _orchestrator(ns).run_all(Path(ns.config), force=ns.force, rerun=ns.rerun)


def cmd_compare(ns: argparse.Namespace) -> None:
    _orchestrator(ns).compare(_output_root_from_args(ns))


def cmd_sync(ns: argparse.Namespace) -> None:
    cfg = Path(ns.config).resolve() if ns.config else None
    _orchestrator(ns).sync(cfg, all_experiments=ns.all, output_root=_output_root_from_args(ns))


def main(argv: list[str] | None = None) -> None:
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
