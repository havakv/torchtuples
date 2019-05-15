
import pytest
import torch
from torchtuples.practical import _accuracy, accuracy_argmax

@pytest.mark.parametrize('input, target, score', [
    ([0, 1, 1], [0, 1, 1], 1),
    ([1, 0, 0], [0, 1, 1], 0),
    ([1, 0, 0, 1], [0, 1, 0, 1], 0.5),
])
def test_accuracy(input, target, score):
    input = torch.tensor(input)
    target = torch.tensor(target)
    acc = _accuracy(input, target)
    assert acc == score

@pytest.mark.parametrize('input, target, score', [
    ([0., 1., 1.], [0, 1, 1], 1),
    ([0.1, 0.6, 0.6], [0, 1, 1], 1),
    ([0.1, 0.1, 0.1, 0.1], [0, 0, 1, 1], 0.5),
    ([[0.1, 0.3, 0.6], [0.3, 0.4, 0.3]], [2, 1], 1),
    ([[0.6, 0.3, 0.1], [0.3, 0.3, 0.4]], [0, 1], 0.5),
])
def test_accuracy_argmax(input, target, score):
    input = torch.tensor(input)
    target = torch.tensor(target)
    acc = accuracy_argmax(input, target)
    assert acc == score