

import pytest
import torch
from torchtuples import optim
from torchtuples import Model
import torchtuples.callbacks as cb

class MocModel:
    def __init__(self, optimizer, batches_per_epoch=None):
        self.optimizer = optimizer
        self.fit_info = dict(batches_per_epoch=batches_per_epoch)


class StopBeforeStep(cb.Callback):
    def before_step(self):
        return True


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
        batches_per_epoch = 2
        model = MocModel(op, batches_per_epoch)
        weight_decay.give_model(model)
        weight_decay.on_fit_start()
        net(inp).mean().backward()
        weight_decay.before_step()
        wd = wd * math.sqrt(1 / (nb_epochs * batches_per_epoch))
        assert (net.weight.data == (weight - wd * weight)).all()

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    @pytest.mark.parametrize('wd', [0.1, 0.01, 0.001])
    def test_decoupled_weight_decay_with_model(self, optim_class, wd):
        torch.manual_seed(1234)
        inp, tar = torch.randn(10, 3), torch.randn(10, 1)
        net = torch.nn.Linear(3, 1)
        weight = net.weight.clone().data
        weight_decay = cb.DecoupledWeightDecay(wd)
        model = Model(net, torch.nn.MSELoss(), optim_class(0.1))
        model.fit(inp, tar, callbacks=[weight_decay, StopBeforeStep()])
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
    def test_decoupled_weight_decay_normalized_with_model(self, optim_class, wd, nb_epochs):
        import math
        torch.manual_seed(1234)
        inp, tar = torch.randn(10, 3), torch.randn(10, 1)
        net = torch.nn.Linear(3, 1)
        weight = net.weight.clone().data
        weight_decay = cb.DecoupledWeightDecay(wd, True, nb_epochs)
        batch_size = 2
        model = Model(net, torch.nn.MSELoss(), optim_class(0.1))
        model.fit(inp, tar, batch_size, callbacks=[weight_decay, StopBeforeStep()])
        batches_per_epoch = model.fit_info['batches_per_epoch']
        wd = wd * math.sqrt(1 / (nb_epochs * batches_per_epoch))
        assert (net.weight.data == (weight - wd * weight)).all()
