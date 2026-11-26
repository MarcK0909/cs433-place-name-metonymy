"""One TRAIN-only optimizer step: a diagnostic, not a scored experiment."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import time

from . import train_encoder as trainer
from .common import data_versions, load_examples, run_info, write_json
from .torch_utils import choose_device, device_name, peak_memory_mb, reset_peak_memory, set_seed, synchronize

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jhu-clsp/mmBERT-base", help="Hub ID or local model directory")
    parser.add_argument("--train", nargs="+", choices=("hipe_fr_train", "hipe_de_train"),
                        default=["hipe_fr_train", "hipe_de_train"])
    parser.add_argument("--data-dir", default="data/processed/group1")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supcon-weight", type=float, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke/encoder.json"))
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if not math.isfinite(args.supcon_weight) or args.supcon_weight < 0:
        parser.error("--supcon-weight must be finite and nonnegative")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("--temperature must be finite and positive")
    if args.supcon_weight > 0 and args.temperature is None:
        parser.error("Positive --supcon-weight requires --temperature")
    if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2"):
        parser.error("Select exactly one allocated GPU 0, 1 or 2 with CUDA_VISIBLE_DEVICES; never GPU 3")
    return args


def run(args) -> dict:
    """Check gradients and an actual parameter update; save no checkpoint."""
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}; choose a new --output")
    device = choose_device(args.device)
    set_seed(args.seed)
    examples = [r for name in args.train for r in load_examples(name, args.data_dir)]
    if len(examples) < args.batch_size:
        raise ValueError("Not enough TRAIN examples for the requested batch")
    batch_examples = random.Random(args.seed).sample(examples, args.batch_size)
    labels = [r["label"] for r in batch_examples]
    if set(labels) != {0, 1}:
        raise ValueError("Sampled batch lacks a class; choose another explicit seed for this diagnostic")
    if args.supcon_weight > 0 and min(labels.count(0), labels.count(1)) < 2:
        raise ValueError("Sampled batch needs at least two examples per class to exercise SupCon")
    print(f"Loading {args.model}; remote models download if not cached.", flush=True)
    tokenizer = trainer.AutoTokenizer.from_pretrained(args.model)
    encoded, shortened = trainer.encode(batch_examples, tokenizer, 512, "mean", 64)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = trainer.MentionClassifier(args.model).to(device)
    model.train()
    batch = trainer.collate(encoded, pad_id, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    fp16 = device.type == "cuda" and not args.no_fp16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    before = model.head.weight.detach().clone()
    reset_peak_memory(device)
    synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss, components = trainer.training_loss(
        model, batch, torch.nn.CrossEntropyLoss(), supcon_weight=args.supcon_weight,
        temperature=args.temperature, fp16=fp16,
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite loss; inspect inputs/precision before training")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, module in (("encoder", model.encoder), ("head", model.head)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError(f"Missing/nonfinite {name} gradients; if using fp16, diagnose with --no-fp16")
        if not any(torch.count_nonzero(g) for g in grads):
            raise ValueError(f"All {name} gradients are zero")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    synchronize(device)
    if torch.equal(before, model.head.weight.detach()):
        raise ValueError("Optimizer did not change the classification head")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite parameters after optimizer step")
    report = {
        "diagnostic_only": True, "optimizer_steps": 1, "train_sets": args.train,
        "sample_ids": [r["id"] for r in batch_examples],
        "class_counts": {"literal": labels.count(0), "metonymic": labels.count(1)},
        "model": args.model, "resolved_model_revision": getattr(model.encoder.config, "_commit_hash", None),
        "device": device_name(device), "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fp16": fp16, "seed": args.seed, "supcon_weight": args.supcon_weight,
        "temperature": args.temperature, "loss": loss.item(), **components,
        "gradient_norm_before_clipping": float(gradient_norm), "head_updated": True,
        "step_seconds": time.perf_counter() - started, "memory_mb": peak_memory_mb(device),
        "memory_measure": "peak CUDA allocated" if device.type == "cuda" else "process peak RSS",
        "context_shortened": shortened, "run_info": run_info(),
        "data_md5": data_versions(args.train, args.data_dir),
        "upstream_commit": trainer.UPSTREAM_COMMIT,
    }
    write_json(args.output, report)
    return report


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    print(f"PASS: one optimizer step; CE={report['ce']:.6f}, SupCon={report['supcon']:.6f}, total={report['loss']:.6f}")
    print(f"Report: {args.output}")
    return report
