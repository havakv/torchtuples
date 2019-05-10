
import pytest
import torch
from torchtuples import optim
from torchtuples.tupletree import TupleTree
from torchtuples.testing import assert_tupletree_equal

def get_params(net):
    return TupleTree(net.parameters()).apply(lambda x: x.detach().clone())


class TestOptimWrap:
    def setup(self):
        self.net = torch.nn.Linear(3, 1)

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    def test_torch_optimizer(self, optimizer_class):
        optimizer = optimizer_class(self.net.parameters(), lr=0.1)
        op = optim.OptimWrap(optimizer)
        assert op.optimizer is optimizer

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    @pytest.mark.parametrize('lr', [0.1, 0.01, 0.5])
    def test_torch_optimizer_lr(self, optimizer_class, lr):
        optimizer = optimizer_class(self.net.parameters(), lr=lr)
        op = optim.OptimWrap(optimizer)
        assert optimizer.param_groups[0]['lr'] == lr
        op.set_lr(0.1234)
        lr = optimizer.param_groups[0]['lr']
        assert lr == 0.1234

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    def test_torch_optimizer_set(self, optimizer_class):
        optimizer = optimizer_class(self.net.parameters(), lr=0.1)
        op = optim.OptimWrap(optimizer)
        op.set('0.1234', 0.1234)
        val = optimizer.param_groups[0]['0.1234']
        assert val == 0.1234

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    def test_zero_grad(self, optimizer_class):
        optimizer = optimizer_class(self.net.parameters(), lr=0.1)
        op = optim.OptimWrap(optimizer)
        inp = torch.randn(10, 3)
        self.net(inp).mean().backward()
        a = TupleTree(op.param_groups[0]['params'])
        assert not a.apply(lambda x: (x.grad == 0.).all()).all()
        op.zero_grad()
        assert a.apply(lambda x: (x.grad == 0.).all()).all()

    @pytest.mark.parametrize('optimizer_class', [
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adam,
        torch.optim.Adagrad,
    ])
    def test_step(self, optimizer_class):
        torch.manual_seed(1234)
        net = torch.nn.Linear(5, 1)
        optimizer = optimizer_class(net.parameters(), lr=0.1)
        op = optim.OptimWrap(optimizer)
        op.zero_grad()
        params = get_params(net)
        inp = torch.randn(10, 5)
        net(inp).mean().backward()
        op.step()
        new_params = get_params(net)
        with pytest.raises(AssertionError) as ex:
            assert_tupletree_equal(params, new_params)
        assert str(ex.value) == "Not equal values"


class TestOptimWrapReinit:
    def setup(self):
        torch.manual_seed(1234)
        # self.input, self.target = torch.randn(10, 3), torch.randn(10)
        self.net = torch.nn.Linear(3, 1)

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    @pytest.mark.parametrize('lr', [0.1, 0.01, 0.5])
    def test_reinitialize(self, optim_class, lr):
        op = optim_class(lr=lr, params=self.net.parameters())
        assert op.param_groups[0]['lr'] == lr
        sd = op.state_dict()
        op.set_lr(0.4321)
        op2 = op.reinitialize(self.net.parameters())
        assert type(op.optimizer) is type(op2.optimizer)
        assert op.state_dict() != sd
        assert op2.state_dict() == sd

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    def test_call(self, optim_class):
        op = optim_class(lr=0.1234)
        op_self = op(self.net.parameters())
        assert op is op_self
        sd = op.state_dict()
        op.set_lr(0.4321)
        op2 = op.reinitialize(self.net.parameters())
        assert type(op.optimizer) is type(op2.optimizer)
        assert op.state_dict() != sd
        assert op2.state_dict() == sd

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    def test_zero_grad(self, optim_class):
        op = optim_class()
        op(self.net.parameters())
        inp = torch.randn(10, 3)
        self.net(inp).mean().backward()
        a = TupleTree(op.param_groups[0]['params'])
        assert not a.apply(lambda x: (x.grad == 0.).all()).all()
        op.zero_grad()
        assert a.apply(lambda x: (x.grad == 0.).all()).all()

    @pytest.mark.parametrize('optim_class', [
        optim.SGD,
        optim.RMSprop,
        optim.Adam,
        optim.AdamW,
        optim.AdamWR,
    ])
    def test_step(self, optim_class):
        torch.manual_seed(1234)
        net = torch.nn.Linear(5, 1)
        op = optim_class(lr=0.1)
        op(net.parameters())
        op.zero_grad()
        params = get_params(net)
        inp = torch.randn(10, 5)
        net(inp).mean().backward()
        op.step()
        new_params = get_params(net)
        with pytest.raises(AssertionError) as ex:
            assert_tupletree_equal(params, new_params)
        assert str(ex.value) == "Not equal values"
