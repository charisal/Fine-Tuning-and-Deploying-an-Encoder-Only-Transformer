import argparse
import os
from datasets import load_dataset
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True, help="Path to train.json or train.jsonl")
    ap.add_argument("--eval_file", default=None, help="Path to eval.json or eval.jsonl")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--output_dir", default="model/finetuned")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    ap.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for Ampere+ GPUs)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    args = ap.parse_args()

    # Load dataset from local JSON/JSONL
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    ds = load_dataset("json", data_files=data_files)

    # Validate schema
    required = {"text", "label"}
    missing = required - set(ds["train"].column_names)
    if missing:
        raise ValueError(f"Train file missing columns: {missing}. Found: {ds['train'].column_names}")

    # Infer num_labels (assumes integer labels)
    labels = sorted(set(ds["train"]["label"]))
    if not all(isinstance(x, (int, np.integer)) for x in labels):
        raise ValueError(f"Labels must be integers for Deberta. Found label types: {set(type(x) for x in labels)}")
    num_labels = int(max(labels)) + 1
    print(f"✓ Detected {num_labels} labels: {labels}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def preprocess(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(preprocess, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("label",)])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    # LoRA config: target common DeBERTa attention projection modules
    # DeBERTa often uses "query_proj" / "key_proj" / "value_proj" / "dense" depending on implementation.
    # We'll start with a conservative set; if PEFT reports "target modules not found", we’ll adjust.
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],    # Where to attach Lora adapters, those are small adjustments
        modules_to_save=["classifier"],  # Which layers to make fully trainable

    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1_macro": f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=50,
        eval_strategy="epoch" if "validation" in ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16 and torch.cuda.is_available(),  # Use BF16 instead of FP16
        fp16=args.fp16 and torch.cuda.is_available() and not args.bf16,  # Only if bf16 not enabled
        max_grad_norm=1.0,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        compute_metrics=compute_metrics if "validation" in ds else None,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
