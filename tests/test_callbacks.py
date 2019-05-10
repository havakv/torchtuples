

import pytest
import torch
from torchtuples import optim
# from torchtuples.tupletree import TupleTree
# from torchtuples.testing import assert_tupletree_equal
import torchtuples.callbacks as cb

class MocModel:
    def __init__(self, optimizer, batches_per_epoch=None):
        self.optimizer = optimizer
        self.fit_info = dict(batches_per_epoch=batches_per_epoch)

class TestDecoupledWeightDecay:
    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    @pytest.mark.parametrize('wd', [0.1, 0.01, 0.001])
    def test_decoupled_weight_decay(self, optim_class, wd):
        torch.manual_seed(1234)
        inp = torch.randn(10, 3)
        net = torch.nn.Linear(3, 1)
        weight = net.weight.clone().data
        weight_decay = cb.DecoupledWeightDecay(wd)
        op = optim_class()(net.parameters())
        model = MocModel(op)
        weight_decay.give_model(model)
        net(inp).mean().backward()
        weight_decay.before_step()
        assert (net.weight.data == (weight - wd * weight)).all()

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    @pytest.mark.parametrize('wd', [0.1, 0.01, 0.001])
    @pytest.mark.parametrize('nb_epochs', [2, 3])
    def test_decoupled_weight_decay_normalized(self, optim_class, wd, nb_epochs):
        import math
        torch.manual_seed(1234)
        inp = torch.randn(10, 3)
        net = torch.nn.Linear(3, 1)
        weight = net.weight.clone().data
        weight_decay = cb.DecoupledWeightDecay(wd, True, nb_epochs)
        op = optim_class()(net.parameters())
        batch_size = 2
        model = MocModel(op, batch_size)
        weight_decay.give_model(model)
        weight_decay.on_fit_start()
        net(inp).mean().backward()
        weight_decay.before_step()
        wd = wd * math.sqrt(1 / (nb_epochs * batch_size))
        assert (net.weight.data == (weight - wd * weight)).all()


