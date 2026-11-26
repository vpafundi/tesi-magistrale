"""Generic Optuna tuning loop for classic predictors."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ..predictors import masked_predictions_from_proba
from .protocols import TunableMethod
from .storage import clear_journal_storage, study_storage


@dataclass
class ClassicEvalData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    seed: int
    macro_scope: str | None = None
    train_df_for_mask: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


@dataclass
class ClassicTuningResult:
    pred: pd.Series
    proba_df: pd.DataFrame | None
    extra_run_info: dict[str, Any]


def _default_tuning_cfg(params: dict[str, Any]) -> dict[str, Any]:
    tuning = dict(params.get("tuning") or {})
    tuning.setdefault("strategy", "cv")
    tuning.setdefault("n_splits", 5)
    tuning.setdefault("n_trials", 50)
    tuning.setdefault("seed", 42)
    tuning.setdefault("refit_on", "train_val")
    tuning.setdefault("storage", None)
    return tuning


def _build_folds(data: ClassicEvalData, tuning: dict[str, Any]) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    strategy = tuning["strategy"]
    if strategy == "val":
        return [(data.train_df, data.val_df, "val")]
    pool = pd.concat([data.train_df, data.val_df], ignore_index=True)
    n_splits = int(tuning["n_splits"])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(tuning["seed"]))
    y = pool["sub_category"]
    groups = pool["document_id"]
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for i, (tr_idx, va_idx) in enumerate(sgkf.split(pool, y, groups=groups)):
        tr = pool.iloc[tr_idx].reset_index(drop=True)
        ev = pool.iloc[va_idx].reset_index(drop=True)
        folds.append((tr, ev, f"fold{i}"))
    return folds


def _preds_from_proba(
    proba: np.ndarray,
    classes: list[str],
    eval_df: pd.DataFrame,
    data: ClassicEvalData,
) -> list[str]:
    mask_df = data.train_df_for_mask if len(data.train_df_for_mask) else data.train_df
    return masked_predictions_from_proba(
        proba,
        classes,
        eval_df,
        macro_scope=data.macro_scope,
        train_df=mask_df,
    )


def tune_and_predict(
    method: TunableMethod,
    data: ClassicEvalData,
    *,
    spec_params: dict[str, Any],
    artifacts_dir: Path,
    slug: str,
    logger: logging.Logger,
    rerun: bool = False,
) -> ClassicTuningResult:
    """Run Optuna study, refit best config, predict on test."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tuning = _default_tuning_cfg(spec_params)
    search_space = dict(spec_params.get("search_space") or {})
    n_trials = int(tuning["n_trials"])
    study_name = f"{slug}__tuning"

    journal_path = artifacts_dir / f"optuna__{slug}.log"
    if rerun:
        clear_journal_storage(journal_path)
        for p in artifacts_dir.glob(f"tuning__{slug}.*"):
            p.unlink(missing_ok=True)

    storage = study_storage(tuning.get("storage"), journal_path=journal_path)
    sampler = TPESampler(seed=int(tuning["seed"]))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    folds = _build_folds(data, tuning)
    logger.info(
        "tuning start method=%s strategy=%s folds=%s n_trials=%s pool_train=%s val=%s test=%s journal=%s",
        getattr(method, "name", type(method).__name__),
        tuning["strategy"],
        len(folds),
        n_trials,
        len(data.train_df),
        len(data.val_df),
        len(data.test_df),
        journal_path,
    )

    trial_durations: list[float] = []
    best_so_far = float("-inf")
    best_trial_num = -1
    t_study = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far, best_trial_num
        t0 = time.perf_counter()
        params = method.suggest_params(trial, search_space)
        fold_scores: list[float] = []

        for fold_i, (tr, ev, fold_key) in enumerate(folds):
            if hasattr(method, "fit_fold_eval"):
                model, x_ev = method.fit_fold_eval(tr, ev, params, fold_key=fold_key)  # type: ignore[attr-defined]
                proba, classes = method.predict_proba_from_x(model, x_ev)  # type: ignore[attr-defined]
            else:
                model = method.fit(tr, params, fold_key=fold_key)
                proba, classes = method.predict_proba(model, ev)
            preds = _preds_from_proba(proba, classes, ev, data)
            score = float(f1_score(ev["sub_category"], preds, average="macro"))
            fold_scores.append(score)
            mean_so_far = float(np.mean(fold_scores))
            logger.info(
                "tuning trial %s/%s fold %s/%s | f1=%.4f | mean=%.4f",
                trial.number + 1,
                n_trials,
                fold_i + 1,
                len(folds),
                score,
                mean_so_far,
            )
            trial.report(mean_so_far, fold_i)
            if trial.should_prune():
                logger.info("tuning trial %s PRUNED at fold %s", trial.number + 1, fold_i + 1)
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
        elapsed_trial = time.perf_counter() - t0
        trial_durations.append(elapsed_trial)
        if mean_score > best_so_far:
            best_so_far = mean_score
            best_trial_num = trial.number
        avg_dur = sum(trial_durations) / len(trial_durations)
        remaining = max(0, n_trials - (trial.number + 1))
        eta = remaining * avg_dur
        logger.info(
            "tuning trial %s/%s COMPLETE | %s | cv_f1=%.4f +/- %.4f | best=%.4f (#%s) | trial %.1fs | ETA ~%.0fs",
            trial.number + 1,
            n_trials,
            params,
            mean_score,
            std_score,
            best_so_far,
            best_trial_num + 1,
            elapsed_trial,
            eta,
        )
        return mean_score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    terminal_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    )
    n_done = len([t for t in study.trials if t.state in terminal_states])
    remaining = max(0, n_trials - n_done)
    if n_done:
        logger.info("tuning resume: %s/%s trials already finished", n_done, n_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    elif not study.trials:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best = study.best_trial
    best_params = dict(best.params)
    if hasattr(method, "finalize_trial_params"):
        best_params = method.finalize_trial_params(best_params)  # type: ignore[attr-defined]
    logger.info(
        "tuning study done in %.1fs | best_trial=%s best_value=%.4f params=%s",
        time.perf_counter() - t_study,
        best.number,
        best.value,
        best_params,
    )

    refit_on = tuning["refit_on"]
    if refit_on == "train_val":
        refit_df = pd.concat([data.train_df, data.val_df], ignore_index=True)
    else:
        refit_df = data.train_df
    if hasattr(method, "refit"):
        final_model = method.refit(refit_df, best_params)  # type: ignore[attr-defined]
    else:
        final_model = method.fit(refit_df, best_params, fold_key="refit")

    proba, classes = method.predict_proba(final_model, data.test_df)
    preds = _preds_from_proba(proba, classes, data.test_df, data)
    pred_series = pd.Series(preds, index=data.test_df.index)

    proba_cols = {f"proba__{c}": proba[:, i] for i, c in enumerate(classes)}
    proba_df = pd.DataFrame(proba_cols, index=data.test_df.index)

    fold_scores_best: list[float] = []
    for tr, ev, fold_key in folds:
        if hasattr(method, "fit_fold_eval"):
            model, x_ev = method.fit_fold_eval(tr, ev, best_params, fold_key=fold_key)  # type: ignore
            p, cl = method.predict_proba_from_x(model, x_ev)  # type: ignore
        else:
            model = method.fit(tr, best_params, fold_key=fold_key)
            p, cl = method.predict_proba(model, ev)
        pr = _preds_from_proba(p, cl, ev, data)
        fold_scores_best.append(float(f1_score(ev["sub_category"], pr, average="macro")))

    tuning_json = {
        "best_params": best_params,
        "cv_macro_f1": float(best.value),
        "cv_macro_f1_std": float(np.std(fold_scores_best)) if fold_scores_best else 0.0,
        "cv_fold_scores": fold_scores_best,
        "n_trials": n_trials,
        "n_completed_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "refit_on": refit_on,
        "strategy": tuning["strategy"],
        "n_splits": tuning.get("n_splits"),
        "best_trial_number": best.number,
    }

    csv_path = artifacts_dir / f"tuning__{slug}.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    json_path = artifacts_dir / f"tuning__{slug}.json"
    json_path.write_text(json.dumps(tuning_json, ensure_ascii=False, indent=2), encoding="utf-8")

    extra = {"tuning": tuning_json, "best_params": best_params}
    return ClassicTuningResult(pred=pred_series, proba_df=proba_df, extra_run_info=extra)