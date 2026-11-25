"""Approach A: fine-tune mmBERT to label each place name literal (0) or metonymic (1).

The encoder reads the passage around the place name (up to 64 HIPE tokens on
each side by default, as in the lab's shared baseline). A linear head on the
place name's representation gives the two classes. Loss: cross-entropy, optionally plus supervised contrastive loss. Representations (--representation):
    mean           mean of the place name's subwords (the lab's Option B)
    first-subword  the place name's first subword (the lab's Option A)
    marker         the text marks the name, "... de [[Paris]] a ...", and the
                   head reads the opening marker (an extra configuration)

With --freeze-encoder this is the probe baseline of the lab slides instead:
mmBERT is not trained, and an L2 logistic regression is fitted on its
features (penalty chosen on dev).

Default settings follow the lab's tuned mmBERT recipe for HIPE-2020 NER
(impresso-ner-training, configs/training.yaml): learning rate 5e-5, cosine
decay after 10% warm-up, batch 32, no weight decay, up to 20 epochs with
early stopping after 6 epochs without a better dev score, seeds 42, 123,
456, 789, 2025.

Run from this repository root, on an allocated GPU (never GPU 3):
    CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --train hipe_fr_train hipe_de_train --dev hipe_fr_dev hipe_de_dev
    CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --train hipe_fr_train hipe_de_train --dev hipe_fr_dev hipe_de_dev --freeze-encoder

Results go to outputs/runs/<run name>/:
    config.json               all settings, versions, machine and data md5
    metrics.json              dev score per epoch, scores per set, time, memory
    predictions_<set>.jsonl   id, label, prob_metonymic, prediction
Test sets are predicted and scored only when named with --final-test.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import common
from .common import data_versions, load_examples, run_info, scores, write_json, write_jsonl
from .passages import BRACKETS, passage
from .torch_utils import (
    choose_device,
    device_name,
    parameter_counts,
    peak_memory_mb,
    reset_peak_memory,
    set_seed,
    synchronize,
)

import torch  # noqa: E402  (after torch_utils, which sets the GPU numbering)
from torch import nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from metonymy.contrastive import supervised_contrastive_loss

UPSTREAM_COMMIT = "d5234ba3110b55101111685e0df9da521ba2c0d7"

PROBE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SHORTER_WINDOWS = (96, 64, 48, 32, 16, 8, 4, 0)  # tried in order when a passage is too long


# ---------- Inputs ----------


def encode(examples, tokenizer, max_length, representation="mean", max_words=None, sentence_only=False):
    """Subword ids, the target's subwords and its pooling position, per example.

    With representation "marker" the text marks the name with [[ ]], and the
    pooling position is the opening marker; otherwise the text is plain and
    the position is the name's first subword. Every subword of the name must
    survive: if a passage is longer than max_length subwords, its context is
    cut to fewer tokens on each side until it fits (the name always stays).
    Returns the items and how many passages were cut.
    """
    markers = BRACKETS if representation == "marker" else None
    items, shortened = [], 0
    for example in examples:
        windows = [max_words] + [w for w in SHORTER_WINDOWS if max_words is None or w < max_words]
        for window in windows:
            text = passage(example, window, markers, sentence_only)
            encoding = tokenizer(text["text"], return_offsets_mapping=True)
            if len(encoding["input_ids"]) <= max_length:
                break
        else:
            raise ValueError(f"{example['id']}: longer than {max_length} subwords even without context")
        shortened += window != max_words
        offsets = encoding["offset_mapping"]
        position = [i for i, (s, e) in enumerate(offsets) if s < e and s <= text["marker_start"] < e]
        mention = [int(s < e and s < text["name_end"] and e > text["name_start"]) for s, e in offsets]
        if not position or not any(mention):
            raise ValueError(f"{example['id']}: the place name was not found in the subwords")
        items.append(
            {
                "id": example["id"],
                "input_ids": encoding["input_ids"],
                "target_index": position[0],
                "mention_mask": mention,
                "label": example["label"],
            }
        )
    return items, shortened


def collate(items, pad_id, device):
    """Pad a list of items into one batch of tensors on the device."""
    length = max(len(item["input_ids"]) for item in items)
    ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), length), dtype=torch.long)
    mention = torch.zeros((len(items), length), dtype=torch.long)
    for row, item in enumerate(items):
        n = len(item["input_ids"])
        ids[row, :n] = torch.tensor(item["input_ids"])
        attention[row, :n] = 1
        mention[row, :n] = torch.tensor(item["mention_mask"])
    return {
        "input_ids": ids.to(device),
        "attention_mask": attention.to(device),
        "mention_mask": mention.to(device),
        "target_index": torch.tensor([item["target_index"] for item in items]).to(device),
        "labels": torch.tensor([item["label"] for item in items]).to(device),
    }


# ---------- Model ----------


class MentionClassifier(nn.Module):
    """mmBERT + a linear head on the place-name representation."""

    def __init__(self, model_name, representation="mean", dropout=0.1, attn="sdpa", compile_model=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "reference_compile"):  # ModernBERT (mmBERT) may otherwise call torch.compile
            config.reference_compile = compile_model
        try:
            self.encoder = AutoModel.from_pretrained(model_name, config=config, attn_implementation=attn)
        except (ValueError, ImportError) as error:
            print(f"attention '{attn}' not available ({error}); using 'eager'")
            self.encoder = AutoModel.from_pretrained(model_name, config=config, attn_implementation="eager")
        self.representation = representation
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def features(self, batch):
        hidden = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        if self.representation == "mean":
            weights = batch["mention_mask"].unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[rows, batch["target_index"]]  # first subword, or the opening marker

    def forward(self, batch):
        return self.head(self.dropout(self.features(batch)))


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def predict(model, items, args, device, fp16, pad_id):
    """Probability of "metonymic" for every item."""
    model.eval()
    # Batches of similar length waste less compute on padding (fair timing).
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    result = np.zeros(len(items))
    for rows in batches(order, args.eval_batch_size):
        batch = collate([items[i] for i in rows], pad_id, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits = model(batch)
        result[rows] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    if not np.isfinite(result).all():  # NaN would silently count as "literal"
        raise FloatingPointError("the model gave NaN probabilities; try --no-fp16")
    return result


@torch.no_grad()
def extract_features(model, items, args, device, fp16, pad_id):
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    features = None
    for rows in batches(order, args.eval_batch_size):
        batch = collate([items[i] for i in rows], pad_id, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            values = model.features(batch).float().cpu().numpy()
        if features is None:
            features = np.zeros((len(items), values.shape[1]), dtype=np.float32)
        features[rows] = values
    return features


# ---------- Training ----------


def macro_f1(labels, probabilities):
    return scores(labels, (probabilities >= 0.5).astype(int))["macro_f1"]


def training_loss(model, batch, loss_function, *, supcon_weight, temperature, fp16):
    """One encoder forward; CE sees original features, SupCon normalized ones.

    Weight zero follows Group 1's original forward and never calls SupCon.
    Returns total loss plus detached diagnostics for epoch logging.
    """
    with torch.autocast(batch["input_ids"].device.type, dtype=torch.float16, enabled=fp16):
        if supcon_weight == 0:
            logits = model(batch)
            features = None
        else:
            features = model.features(batch)
            logits = model.head(model.dropout(features))
    ce = loss_function(logits.float(), batch["labels"])
    cl = ce.new_zeros(())
    eligible = 0
    if supcon_weight > 0:
        cl = supervised_contrastive_loss(features.float(), batch["labels"], temperature=temperature)
        labels = batch["labels"]
        eligible = int(((labels[:, None] == labels[None, :]).sum(1) > 1).sum().item())
    total = ce if supcon_weight == 0 else ce + supcon_weight * cl
    return total, {"ce": ce.item(), "supcon": cl.item(), "eligible_anchors": eligible,
                   "anchors": len(batch["labels"])}


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
        ce_losses, cl_losses, eligible_anchors, anchors = [], [], 0, 0
        order = order_rng.permutation(len(train_items))
        for chunk in batches([train_items[i] for i in order], args.batch_size):
            batch = collate(chunk, pad_id, device)
            loss, components = training_loss(
                model, batch, loss_function, supcon_weight=args.supcon_weight,
                temperature=args.temperature, fp16=fp16,
            )
            ce_losses.append(components["ce"])
            cl_losses.append(components["supcon"])
            eligible_anchors += components["eligible_anchors"]
            anchors += components["anchors"]
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
                        "ce_loss": float(np.mean(ce_losses)), "supcon_loss": float(np.mean(cl_losses)),
                        "weighted_supcon_loss": args.supcon_weight * float(np.mean(cl_losses)),
                        "supcon_eligible_anchors": eligible_anchors,
                        "supcon_anchors_without_positive": anchors - eligible_anchors if args.supcon_weight > 0 else 0,
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


def fit_probe(train_x, train_y, dev_x, dev_y):
    """L2 logistic regression on frozen features; the penalty is chosen on dev."""
    from .probe import fit_logistic

    mean, scale = train_x.mean(0), train_x.std(0) + 1e-6
    standard = lambda x: (x - mean) / scale  # noqa: E731
    best, history = None, []
    for lambda_ in PROBE_LAMBDAS:
        w = fit_logistic(standard(train_x), train_y, lambda_)
        f1 = macro_f1(dev_y, expit(standard(dev_x) @ w[1:] + w[0]))
        history.append({"lambda": lambda_, "dev_macro_f1": f1})
        print(f"probe, lambda {lambda_:g}: dev macro-F1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, lambda_, w)
    _, lambda_, w = best
    return (lambda x: expit(standard(x) @ w[1:] + w[0])), lambda_, history


# ---------- Main ----------


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="jhu-clsp/mmBERT-base", help="Hugging Face id or local folder")
    parser.add_argument("--tokenizer", help="default: the same as --model")
    parser.add_argument("--train", nargs="+", required=True, help="training sets, e.g. hipe_fr_train")
    parser.add_argument("--dev", nargs="+", required=True,
                        help="sets for every choice (epoch, penalty), pooled, e.g. hipe_fr_dev hipe_de_dev")
    parser.add_argument("--eval", nargs="*", default=[], help="extra sets to predict (not test sets)")
    parser.add_argument("--final-test", nargs="*", default=[], help="test sets: only for the final evaluation")
    parser.add_argument("--freeze-encoder", action="store_true", help="probe baseline: frozen mmBERT + logistic regression")
    parser.add_argument("--representation", choices=["mean", "first-subword", "marker"], default="mean",
                        help="mean = lab Option B, first-subword = lab Option A, marker = [[ ]] in the text")
    parser.add_argument("--max-words", type=int, default=64,
                        help="HIPE tokens of context on each side (lab's shared baseline: 64)")
    parser.add_argument("--context", choices=["window", "sentence"], default="window",
                        help="window = up to --max-words tokens; sentence = only the target's sentence (within it)")
    parser.add_argument("--max-length", type=int, default=512, help="tokens per passage")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--schedule", choices=["cosine", "linear"], default="cosine", help="decay after the warm-up")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6, help="stop after this many epochs without a better dev score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=float, default=0.1, help="share of steps with a rising learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42, help="the lab uses 42, 123, 456, 789, 2025")
    parser.add_argument("--max-train", type=int, help="use only this many training mentions (quick pilot)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--no-fp16", action="store_true", help="on CUDA, train in fp32 instead of fp16")
    parser.add_argument("--attn", default="sdpa", help="attention implementation (sdpa, eager)")
    parser.add_argument("--compile", action="store_true", help="let ModernBERT use torch.compile (off by default)")
    parser.add_argument("--data-dir", help="default: data/processed")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", help="default: outputs/runs/<run name>")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="also save the weights (1.2 GB for mmBERT-base)")
    parser.add_argument("--dry-run", action="store_true", help="load and encode the data, one forward pass, no training")
    parser.add_argument("--supcon-weight", type=float, default=0.0,
                        help="lambda in CE + lambda * SupCon; zero reproduces CE")
    parser.add_argument("--temperature", type=float,
                        help="positive SupCon temperature; required when lambda > 0")
    args = parser.parse_args(argv)
    if not math.isfinite(args.supcon_weight) or args.supcon_weight < 0:
        parser.error("--supcon-weight must be finite and nonnegative")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("--temperature must be finite and positive")
    if args.supcon_weight > 0 and (args.temperature is None or args.freeze_encoder):
        parser.error("SupCon requires --temperature and full fine-tuning (no --freeze-encoder)")
    if any(name.endswith("_test") for name in args.train + args.dev + args.eval):
        parser.error("TEST sets are allowed only through --final-test")
    if set(args.train) & set(args.dev + args.eval + args.final_test):
        parser.error("TRAIN and evaluation sets must be distinct")
    return args


def default_run_name(args):
    model = Path(args.model.rstrip("/")).name
    train = "+".join(args.train)
    if args.freeze_encoder:
        setting = f"probe_{args.representation}"
    else:
        setting = f"lr{args.lr:g}_ep{args.epochs}_{args.representation}"
        if args.schedule != "cosine":
            setting += f"_{args.schedule}"
    setting += f"_w{args.max_words}" if args.max_words is not None else "_wall"
    if args.context == "sentence":
        setting += "_sentence"
    if args.class_weights != "none":
        setting += "_cw"
    if args.max_train:  # pilot runs on part of the data
        setting += f"_n{args.max_train}"
    if args.supcon_weight > 0:
        setting += f"_supcon{args.supcon_weight:g}_tau{args.temperature:g}"
    if args.final_test:  # final runs never overwrite the dev runs
        setting += "_final"
    return f"encoder_{model}_{train}_{setting}_seed{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args.device)
    fp16 = device.type == "cuda" and not args.no_fp16
    set_seed(args.seed)
    run_name = args.run_name or default_run_name(args)
    output = Path(args.output_dir or common.RESULTS / "runs" / run_name)
    if not args.dry_run and output.exists() and any(output.iterdir()) and not args.overwrite:
        sys.exit(f"{output} already has results; use --overwrite or another --run-name")

    train = [example for name in args.train for example in load_examples(name, args.data_dir)]
    if args.max_train and args.max_train < len(train):
        train = random.Random(args.seed).sample(train, args.max_train)
    sets = {name: load_examples(name, args.data_dir) for name in [*args.dev, *args.eval, *args.final_test]}
    print(f"run {run_name} on {device_name(device)}; train {len(train)} mentions; sets {', '.join(sets)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    options = (args.representation, args.max_words, args.context == "sentence")
    train_items, train_cut = encode(train, tokenizer, args.max_length, *options)
    encoded, cut = {}, {"train": train_cut}
    for name, examples in sets.items():
        encoded[name], cut[name] = encode(examples, tokenizer, args.max_length, *options)
    lengths = {"train": [len(item["input_ids"]) for item in train_items]}
    lengths.update({name: [len(item["input_ids"]) for item in items] for name, items in encoded.items()})
    token_stats = {
        name: {"mean": float(np.mean(values)), "max": int(np.max(values)), "context_cut": cut[name]}
        for name, values in lengths.items()
    }
    for name, stats in token_stats.items():
        print(f"tokens {name}: mean {stats['mean']:.0f}, max {stats['max']}, context cut for {stats['context_cut']}")

    model = MentionClassifier(args.model, args.representation, args.dropout, args.attn, args.compile).to(device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    if args.dry_run:
        batch = collate(train_items[: args.batch_size], pad_id, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits = model(batch)
        print(f"dry run: one batch {tuple(batch['input_ids'].shape)} -> logits {tuple(logits.shape)}; nothing trained")
        return

    reset_peak_memory(device)
    started = time.perf_counter()
    dev_items = [item for name in args.dev for item in encoded[name]]  # choices on all dev sets together
    dev_labels = np.array([item["label"] for item in dev_items])
    summary = {
        "system": "encoder-probe" if args.freeze_encoder else "encoder-finetuned",
        "objective": "ce+supcon" if args.supcon_weight > 0 else "ce",
        "upstream_commit": UPSTREAM_COMMIT,
        "representation": args.representation,
        "context": f"{args.context}, up to {args.max_words} tokens each side",
    }
    if args.freeze_encoder:
        train_x = extract_features(model, train_items, args, device, fp16, pad_id)
        dev_x = extract_features(model, dev_items, args, device, fp16, pad_id)
        probe, lambda_, history = fit_probe(train_x, np.array([i["label"] for i in train_items]), dev_x, dev_labels)
        summary.update({"probe_lambda": lambda_, "history": history})
    else:
        history, best_epoch = fine_tune(model, train_items, dev_items, args, device, fp16, pad_id)
        summary.update({"best_epoch": best_epoch, "history": history})
    synchronize(device)
    summary["train_seconds"] = time.perf_counter() - started

    summary["sets"] = {}
    for name, items in encoded.items():
        started = time.perf_counter()
        if args.freeze_encoder:
            probabilities = probe(extract_features(model, items, args, device, fp16, pad_id))
        else:
            probabilities = predict(model, items, args, device, fp16, pad_id)
        synchronize(device)
        seconds = time.perf_counter() - started
        labels = np.array([item["label"] for item in items])
        predictions = (probabilities >= 0.5).astype(int)
        write_jsonl(
            output / f"predictions_{name}.jsonl",
            [
                {"id": item["id"], "label": item["label"], "prob_metonymic": round(float(p), 6), "prediction": int(p >= 0.5)}
                for item, p in zip(items, probabilities)
            ],
        )
        summary["sets"][name] = {
            **scores(labels, predictions),
            "mentions": len(items),
            "metonymic": int(labels.sum()),
            "final_test": name in args.final_test,
            "predict_seconds": seconds,
            "ms_per_mention": 1000 * seconds / len(items),
        }
        print(f"{name}: macro-F1 {summary['sets'][name]['macro_f1']:.4f} ({1000 * seconds / len(items):.1f} ms per mention)")
    counts = parameter_counts(model)
    if args.freeze_encoder:  # only the logistic regression is trained
        counts["trainable"] = int(train_x.shape[1] + 1)
    summary.update(
        {
            "peak_memory_mb": peak_memory_mb(device),
            "device": device_name(device),
            "fp16": fp16,
            "parameters": counts,
            "tokens": token_stats,
        }
    )
    write_json(output / "metrics.json", summary)
    write_json(
        output / "config.json",
        {**vars(args), "upstream_commit": UPSTREAM_COMMIT, "run_name": run_name, "run_info": run_info(), "data_md5": data_versions([*args.train, *sets], args.data_dir)},
    )
    if args.save_model:
        torch.save(model.state_dict(), output / "model.pt")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
