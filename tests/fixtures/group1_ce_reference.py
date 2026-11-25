"""Unmodified Group 1 CE loop from d5234ba; AGPL-3.0 (see group1/LICENSE).

Regression oracle only. Dependencies are the same retained upstream helpers.
"""
import math
import time
import sys
import numpy as np
import torch
from torch import nn
from metonymy.group1.train_encoder import (
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
    batches, collate, synchronize, macro_f1, predict,
)

def fine_tune(model, train_items, dev_items, args, device, fp16, pad_id):
    """Train the whole model; keep the epoch with the best dev macro-F1."""
    no_decay = ("bias", "norm")
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in named if not any(k in n.lower() for k in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in named if any(k in n.lower() for k in no_decay)], "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = math.ceil(len(train_items) / args.batch_size) * args.epochs
    make_schedule = get_cosine_schedule_with_warmup if args.schedule == "cosine" else get_linear_schedule_with_warmup
    scheduler = make_schedule(optimizer, int(args.warmup * total_steps), total_steps)
    weight = None
    if args.class_weights == "balanced":
        counts = np.bincount([item["label"] for item in train_items], minlength=2)
        weight = torch.tensor(len(train_items) / (2 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    order_rng = np.random.default_rng(args.seed)
    dev_labels = np.array([item["label"] for item in dev_items])
    best_f1, best_epoch, best_state, waited, history = -1.0, 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        losses, steps, skipped = [], 0, 0
        order = order_rng.permutation(len(train_items))
        for chunk in batches([train_items[i] for i in order], args.batch_size):
            batch = collate(chunk, pad_id, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                logits = model(batch)
            loss = loss_function(logits.float(), batch["labels"])
            if not math.isfinite(loss.item()):
                sys.exit(f"epoch {epoch}: the loss is not finite (NaN or inf). Try --no-fp16 or a lower --lr.")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            skipped += fp16 and scaler.get_scale() < scale  # the scaler skipped this step (overflow)
            scheduler.step()
            losses.append(loss.item())
        if skipped > steps / 2:
            sys.exit(f"epoch {epoch}: {skipped} of {steps} steps skipped by fp16 overflow. Try --no-fp16.")
        synchronize(device)
        seconds = time.perf_counter() - started
        f1 = macro_f1(dev_labels, predict(model, dev_items, args, device, fp16, pad_id))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_macro_f1": f1,
                        "train_seconds": seconds, "skipped_steps": int(skipped)})
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}, dev macro-F1 {f1:.4f} ({seconds:.0f} s, {skipped} skipped steps)")
        if f1 > best_f1:
            best_f1, best_epoch, waited = f1, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                print(f"no better dev score for {args.patience} epochs: stop")
                break
    model.load_state_dict(best_state)
    return history, best_epoch

