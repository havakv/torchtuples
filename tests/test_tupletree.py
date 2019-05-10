import pytest
import numpy as np
import torch
from torchtuples.tupletree import TupleTree, tuplefy
from torchtuples.testing import assert_tupletree_equal


@pytest.mark.parametrize('inp', [1, (1, 2), ['a', 2], [1, (1, 2)]])
def test_tuplefy_type(inp):
    t = tuplefy(inp)
    assert type(t) is TupleTree

@pytest.mark.parametrize('inp, types', [([1], tuple), ((1,), list)])
def test_tuplefy_not_list(inp, types):
    t = tuplefy(inp, types=types)
    assert type(t[0]) is type(inp)


class TestTupleTree:
    def test_apply(self):
        a = (1, (2, 3), (4, (5,)))
        a = tuplefy(a)
        b = (2, (4, 6), (8, (10,)))
        assert a.apply(lambda x: x*2) == b

    def test_reduce(self):
        a = ((1, (2, 3), 4),
             (1, (2, 3), 4),
             (1, (2, 3), 4),)
        b = (3, (6, 9), 12)
        a = tuplefy(a)
        assert a.reduce(lambda x, y: x+y) == b

    def test_levels(self):
        a = (1, (2, 3), (4, (5,)))
        a = tuplefy(a)
        assert a.levels == (0, (1, 1), (1, (2,)))

    def test_types(self):
        a = ('a', (1, 4.5), {})
        b = (str, (int, float), dict)
        a = tuplefy(a)
        assert a.types() == b

    def test_numerate(self):
        a = (1, (2, 3), (4, (5,)))
        order = (0, (1, 2), (3, (4,)))
        a = tuplefy(a)
        assert a.numerate() == order

    def test_enumerate(self):
        a = tuplefy(list('abcd'), list('ef'))
        a = tuplefy((('a', 'b', 'c', 'd'), ('e', 'f')))
        assert a.enumerate() == (([0, 'a'], [1, 'b'], [2, 'c'], [3, 'd']), ([4, 'e'], [5, 'f']))

    def test_reorder(self):
        a = tuplefy((('a', 'b', 'c', 'd'), ('e', 'f')))
        order = (0, (1, 2,), (5,))
        assert a.reorder(order)  ==  ('a', ('b', 'c'), ('f',))

    def test_repeat(self):
        a = (1, (2, 3))
        b = ((1, (2, 3)), (1, (2, 3)), (1, (2, 3)))
        assert tuplefy(a).repeat(3) == b

    @pytest.mark.parametrize('a, expected',
        [([1, 1, (1,)], 'default'), ([10, 10,], 10), ([(10, 1), (10, 1)], (10, 1))])
    def test_get_if_all_equal(self, a, expected):
        out = tuplefy(a).get_if_all_equal('default')
        assert out == expected

    def test_zip_leaf(self):
        a = (('a1', ('a2', 'a3')), ('b1', ('b2', 'b3')))
        a = tuplefy(a)
        b = (['a1', 'b1'], (['a2', 'b2'], ['a3', 'b3']))
        assert a.zip_leaf() == b

    def test_unzip_leaf(self):
        a = (('a1', ('a2', 'a3')), ('b1', ('b2', 'b3')))
        b = (['a1', 'b1'], (['a2', 'b2'], ['a3', 'b3']))
        b = tuplefy(b, types=tuple)
        assert b.unzip_leaf() == a

    def test_zip_unzip_leaf(self):
        a = (('a1', ('a2', 'a3')), ('b1', ('b2', 'b3')))
        a = tuplefy(a)
        assert a.zip_leaf().unzip_leaf() == a

    def test_np2torch2np(self):
        np.random.seed(123)
        a = [np.random.normal(size=(i, j)).astype('float32') for i, j in [(2, 3), (1, 2)]]
        a = tuplefy(a, np.random.choice(10, size=(4,)))
        b = a.to_tensor().to_numpy()
        assert_tupletree_equal(a, b)
        # assert a.numerate() == b.numerate()
        # assert a.dtypes() == b.dtypes()
        # assert tuplefy(a, b).reduce(lambda x, y: (x == y).all()) == ((True, True), True)

    def test_torch2np2torch(self):
        torch.manual_seed(123)
        a = [torch.randn((i, j)) for i, j in [(2, 3), (1, 2)]]
        a = tuplefy(a, torch.randint(10,(4,)))
        b = a.to_numpy().to_tensor()
        assert_tupletree_equal(a, b)
        # assert a.numerate() == b.numerate()
        # assert a.dtypes() == b.dtypes()
        # assert tuplefy(a, b).reduce(lambda x, y: (x == y).all()) == ((True, True), True)

    def test_iloc(self):
        torch.manual_seed(123)
        a = [torch.randn((i, j)) for i, j in [(5, 3), (5, 2)]]
        a = tuplefy(a, torch.randint(10,(5,)))
        b = tuplefy((a[0][0][:2], a[0][1][:2]), a[1][:2])
        a = a.iloc[:2]
        assert_tupletree_equal(a, b)
        # assert a.numerate() == b.numerate()
        # assert a.dtypes() == b.dtypes()
        # assert tuplefy(a, b).reduce(lambda x, y: (x == y).all()) == ((True, True), True)

    def test_add_root(self):
        a = tuplefy(1, 2, 3)
        assert a == a.add_root()[0]



