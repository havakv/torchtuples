import pytest
import torch
from torch import nn
from torchtuples import optim, Model, TupleTree, tuplefy
from torchtuples.testing import assert_tupletree_equal


class TestModel:
    def setup(self):
        torch.manual_seed(1234)
        self.net = torch.nn.Linear(10, 3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        self.data = tuplefy(x, y)

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

    def test_score_in_batches(self):
        model = Model(self.net, torch.nn.MSELoss())
        a = model.score_in_batches(*self.data)
        b = model.score_in_batches(self.data)
        assert a == b


class _PredSigmoidNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input):
        return self.net(input)

    def predict(self, input):
        return torch.sigmoid(self.net(input))

class TestPrediction:
    def setup(self):
        torch.manual_seed(1234)
        self.net = torch.nn.Linear(5, 3)
        self.prednet = _PredSigmoidNet(self.net)
        x = torch.randn(4, 5)
        self.data = x
        self.dataloader = tuplefy(x).make_dataloader(5, False)
        self.net_dropout = nn.Sequential(self.net, nn.Dropout(0.5))

    @pytest.mark.parametrize('dl', (True, False))
    @pytest.mark.parametrize('numpy', (True, False, None))
    def test_predict_with_fun(self, dl, numpy):
        data = self.dataloader if dl else self.data
        model = Model(self.net, nn.MSELoss())
        model_pred = Model(self.prednet, nn.MSELoss())
        a = model.predict(data, func=torch.sigmoid, numpy=numpy)
        b = model_pred.predict(data, numpy=numpy)
        a_net = model.predict_net(data, func=torch.sigmoid, numpy=numpy)
        b_net = model_pred.predict_net(data, func=torch.sigmoid, numpy=numpy)
        assert (a == b).all()
        assert (a_net == b_net).all()
        assert (a == a_net).all()

    @pytest.mark.parametrize('dl', (True, False))
    @pytest.mark.parametrize('numpy', (True, False, None))
    def test_predict_no_fun(self, dl, numpy):
        data = self.dataloader if dl else self.data
        model = Model(self.net, nn.MSELoss())
        model_pred = Model(self.prednet, nn.MSELoss())
        a = model.predict(data, numpy=numpy)
        b = model_pred.predict(data, numpy=numpy)
        a_net = model.predict_net(data, numpy=numpy)
        b_net = model_pred.predict_net(data, numpy=numpy)
        assert not (a == b).all()
        assert (a_net == b_net).all()
        assert (a == a_net).all()

    @pytest.mark.parametrize('eval_', (True, False))
    def test_predict_net_eval_(self, eval_):
        torch.manual_seed(0)
        model = Model(self.net_dropout)
        assert model.net.training is True
        pred = model.predict(self.data, eval_=eval_)
        assert model.net.training is True
        pred2 = model.predict(self.data, eval_=eval_)
        assert model.net.training is True
        assert (pred == pred2).all().item() is eval_


