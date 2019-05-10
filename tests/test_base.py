import pytest
import torch
from torchtuples import optim, Model, TupleTree
from torchtuples.testing import assert_tupletree_equal


class TestModel:
    def setup(self):
        torch.manual_seed(1234)
        self.net = torch.nn.Linear(10, 3)

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    @pytest.mark.parametrize('lr', [0.1, 10, 0.5])
    def test_set_optimizer(self, optim_class, lr):
        model = Model(self.net, None, optim_class(lr))
        assert type(model.optimizer) is optim_class
        assert model.optimizer.param_groups[0]['lr'] == lr
        model_no_lr = Model(self.net, None, optim_class)
        assert type(model_no_lr.optimizer) is optim_class

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    @pytest.mark.parametrize('lr', [0.1, 10, 0.5])
    def test_set_torch_optimizer(self, optimizer_class, lr):
        optimizer = optimizer_class(self.net.parameters(), lr=lr)
        model = Model(self.net, None, optimizer)
        assert type(model.optimizer) is optim.OptimWrap
        assert type(model.optimizer.optimizer) is optimizer_class
        assert model.optimizer.param_groups[0]['lr'] == lr
