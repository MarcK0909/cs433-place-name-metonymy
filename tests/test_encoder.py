"""CPU integration on synthetic examples and a tiny local random BERT."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from torch import nn
from metonymy.group1 import train_encoder as trainer


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyMentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(20, 8)
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(8, 2)
        self.calls = 0

    def features(self, batch):
        self.calls += 1
        return self.encoder(batch["input_ids"]).mean(1)

    def forward(self, batch):
        return self.head(self.dropout(self.features(batch)))


def items(prefix="train"):
    return [{"id": f"{prefix}-{i}", "input_ids": [i + 1, i + 2],
             "mention_mask": [1, 1], "target_index": 0, "label": i % 2}
            for i in range(8)]


def arguments(extra=()):
    return trainer.parse_args(["--train", "hipe_fr_train", "--dev", "hipe_fr_dev", *extra])


def test_weight_zero_preserves_loss_and_gradients(monkeypatch):
    torch.manual_seed(42)
    model = TinyMentionModel()
    other = copy.deepcopy(model)
    batch = trainer.collate(items()[:4], 0, torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(7)
    expected = criterion(model(batch).float(), batch["labels"])
    expected.backward()
    def forbidden(*args, **kwargs):
        raise AssertionError("CE must not call SupCon")
    monkeypatch.setattr(trainer, "supervised_contrastive_loss", forbidden)
    torch.manual_seed(7)
    actual, stats = trainer.training_loss(other, batch, criterion, supcon_weight=0, temperature=None, fp16=False)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert other.calls == 1
    assert stats["supcon"] == 0


def test_supcon_has_one_forward_and_changes_encoder_gradient():
    torch.manual_seed(3)
    model = TinyMentionModel()
    ce_model = copy.deepcopy(model)
    batch = trainer.collate(items()[:4], 0, torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(8)
    loss, stats = trainer.training_loss(model, batch, criterion, supcon_weight=.2, temperature=.1, fp16=False)
    loss.backward()
    torch.manual_seed(8)
    ce, _ = trainer.training_loss(ce_model, batch, criterion, supcon_weight=0, temperature=None, fp16=False)
    ce.backward()
    assert model.calls == 1
    assert loss.item() == pytest.approx(stats["ce"] + .2 * stats["supcon"])
    assert stats["eligible_anchors"] == 4
    assert torch.isfinite(model.encoder.weight.grad).all()
    assert not torch.allclose(model.encoder.weight.grad, ce_model.encoder.weight.grad)
    torch.testing.assert_close(model.head.weight.grad, ce_model.head.weight.grad)


def test_complete_ce_loop_matches_pinned_group1_reference():
    path = Path(__file__).parent / "fixtures/group1_ce_reference.py"
    spec = importlib.util.spec_from_file_location("ce_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    args = arguments(["--epochs", "2", "--batch-size", "4", "--device", "cpu"])
    torch.manual_seed(42)
    original = TinyMentionModel()
    adapted = copy.deepcopy(original)
    torch.manual_seed(123)
    old_history, old_best = reference.fine_tune(original, items(), items("dev"), args, torch.device("cpu"), False, 0)
    torch.manual_seed(123)
    new_history, new_best = trainer.fine_tune(adapted, items(), items("dev"), args, torch.device("cpu"), False, 0)
    assert old_best == new_best
    for old, new in zip(old_history, new_history):
        assert old["train_loss"] == new["train_loss"]
        assert old["dev_macro_f1"] == new["dev_macro_f1"]
    for key in original.state_dict():
        torch.testing.assert_close(original.state_dict()[key], adapted.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("extra", [
    ["--supcon-weight", "-1"], ["--supcon-weight", "nan"],
    ["--supcon-weight", ".1"], ["--temperature", "0"],
    ["--supcon-weight", ".1", "--temperature", ".1", "--freeze-encoder"],
    ["--eval", "hipe_fr_test"], ["--dev", "hipe_fr_train"],
])
def test_invalid_cli_rejected(extra):
    with pytest.raises(SystemExit):
        arguments(extra)


def test_run_names_separate_objectives_and_temperatures():
    names = [trainer.default_run_name(arguments(extra)) for extra in [
        [], ["--supcon-weight", ".1", "--temperature", ".1"],
        ["--supcon-weight", ".2", "--temperature", ".1"],
        ["--supcon-weight", ".1", "--temperature", ".2"],
    ]]
    assert len(set(names)) == 4


@pytest.fixture
def local_model_and_data(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "in", "Paris", "New", "York", "said", "."]
    vocab = {word: index for index, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    model_dir = tmp_path / "tiny-model"
    fast.save_pretrained(model_dir)
    torch.manual_seed(42)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32)).save_pretrained(model_dir)
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "dev"):
        examples = [{"id": f"{split}-{i}", "tokens": ["in", "New", "York", "."],
                     "start": 1, "end": 3, "space_after": [True, True, False, False],
                     "label": i % 2, "group": f"{split}-doc-{i}"} for i in range(8)]
        (data / f"hipe_fr_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
    return model_dir, data, examples, fast


@pytest.mark.parametrize("weight", [0., .2])
def test_real_transformer_cpu_training_and_output(tmp_path, local_model_and_data, weight):
    model_dir, data, examples, tokenizer = local_model_and_data
    encoded, cut = trainer.encode(examples, tokenizer, 32, "mean", 64)
    target = [token for token, keep in zip(tokenizer.convert_ids_to_tokens(encoded[0]["input_ids"]),
                                          encoded[0]["mention_mask"]) if keep]
    assert target == ["New", "York"]
    assert cut == 0
    output = tmp_path / "run"
    trainer.main(["--model", str(model_dir), "--data-dir", str(data),
                  "--train", "hipe_fr_train", "--dev", "hipe_fr_dev",
                  "--output-dir", str(output), "--device", "cpu", "--epochs", "2",
                  "--batch-size", "4", "--supcon-weight", str(weight), "--temperature", ".1"])
    config = json.loads((output / "config.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert config["supcon_weight"] == weight
    assert config["upstream_commit"] == trainer.UPSTREAM_COMMIT
    assert metrics["objective"] == ("ce+supcon" if weight else "ce")
    assert len(metrics["history"]) == 2
    for row in metrics["history"]:
        assert row["train_loss"] == pytest.approx(row["ce_loss"] + weight * row["supcon_loss"])
    rows = [json.loads(line) for line in (output / "predictions_hipe_fr_dev.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == {r["id"] for r in examples}
    assert all(0 <= r["prob_metonymic"] <= 1 for r in rows)
    assert not list(output.glob("*test*"))
