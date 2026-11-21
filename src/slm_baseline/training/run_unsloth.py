"""CLI training LoRA con Unsloth (subprocess da ``cli train``)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m slm_baseline.training.run_unsloth")
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--metrics-out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)

    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Dipendenze training mancanti (unsloth, trl, datasets). "
            "Su SageMaker: uv sync --no-default-groups --group train --group sagemaker"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_ds = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_ds = None
    if args.val_jsonl and Path(args.val_jsonl).is_file():
        eval_ds = load_dataset("json", data_files=str(args.val_jsonl), split="train")

    def _format(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = train_ds.map(_format)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_format)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )
    train_result = trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    payload = {
        "base_model": args.base_model,
        "adapter_dir": str(out_dir),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "epochs": args.epochs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])