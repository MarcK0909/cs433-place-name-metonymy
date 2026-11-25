"""Small mathematical/gradient checks; no pretrained models or real data."""
import math

import pytest

torch = pytest.importorskip("torch")

from metonymy.contrastive import supervised_contrastive_loss


def test_matches_hand_calculation_and_excludes_self():
    # Each anchor has one positive with similarity 1 and two negatives with 0.
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(features, labels, temperature=1.)
    assert loss.item() == pytest.approx(math.log(math.e + 2) - 1)


def test_better_class_separation_has_lower_loss():
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    aligned = supervised_contrastive_loss(features, torch.tensor([0, 0, 1, 1]), temperature=.2)
    mixed = supervised_contrastive_loss(features, torch.tensor([0, 1, 0, 1]), temperature=.2)
    assert aligned < mixed


def test_vector_scale_and_batch_order_do_not_change_loss():
    features = torch.tensor([[1., 2.], [2., 1.], [-1., 2.], [-2., 1.]])
    labels = torch.tensor([0, 0, 1, 1])
    expected = supervised_contrastive_loss(features, labels, temperature=.3)
    order = torch.tensor([2, 0, 3, 1])
    scaled = features * torch.tensor([[2.], [3.], [4.], [5.]])
    actual = supervised_contrastive_loss(scaled[order], labels[order], temperature=.3)
    assert actual.item() == pytest.approx(expected.item())


def test_singleton_anchor_is_excluded_but_remains_in_denominator():
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    loss = supervised_contrastive_loss(features, torch.tensor([0, 0, 1]), temperature=1.)
    # Only two valid anchors; each has one positive and one negative.
    assert loss.item() == pytest.approx(math.log(math.e + 1) - 1)


@pytest.mark.parametrize("labels", [[0], [0, 1]])
def test_no_positive_returns_differentiable_zero(labels):
    features = torch.ones((len(labels), 3), requires_grad=True)
    loss = supervised_contrastive_loss(features, torch.tensor(labels), temperature=.1)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(features.grad, torch.zeros_like(features))


def test_all_same_class_averages_over_all_other_positives():
    features = torch.ones((3, 2))
    loss = supervised_contrastive_loss(features, torch.tensor([0, 0, 0]), temperature=.1)
    assert loss.item() == pytest.approx(math.log(2))


def test_gradients_match_numerical_derivatives():
    features = torch.tensor([[1., .2], [.3, 1.], [-1., .4], [.5, -1.]],
                            dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    assert torch.autograd.gradcheck(
        lambda values: supervised_contrastive_loss(values, labels, temperature=.3),
        (features,),
    )
    supervised_contrastive_loss(features, labels, temperature=.3).backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_half_precision_is_promoted_and_stable():
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]],
                            dtype=torch.float16, requires_grad=True)
    loss = supervised_contrastive_loss(features, torch.tensor([0, 0, 1, 1]), temperature=.01)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        supervised_contrastive_loss(torch.ones(2, 3), torch.tensor([0, 0]), temperature=temperature)


@pytest.mark.parametrize("labels", [torch.tensor([0, 2]), torch.tensor([0., 1.]),
                                    torch.tensor([True, False]), torch.tensor([0])])
def test_invalid_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        supervised_contrastive_loss(torch.ones(2, 3), labels, temperature=.1)


@pytest.mark.parametrize("features", [torch.ones(2), torch.empty(0, 3),
                                      torch.ones(2, 3, dtype=torch.int64),
                                      torch.tensor([[float("nan")], [1.]])])
def test_invalid_features(features):
    with pytest.raises(ValueError, match="features"):
        supervised_contrastive_loss(features, torch.tensor([0, 0]), temperature=.1)
