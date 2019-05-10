import pytest
import numpy as np
import torch
from torchtuples.tupletree import tuplefy
from torchtuples.data import DatasetTuple, DataLoaderSlice
from torchtuples.testing import assert_tupletree_equal

class TestDatasetTuple:
    @pytest.mark.parametrize('n', [3, 10, 28])
    def test_len(self, n):
        torch.manual_seed(123)
        a = ((torch.randn(n, 3), torch.randint(200, (n, 2))), torch.randn(n))
        ds = DatasetTuple(*a)
        assert len(ds) == n

    @pytest.mark.parametrize('n', [3, 10, 28])
    def test_len_error(self, n):
        torch.manual_seed(123)
        a = ((torch.randn(5, 3), torch.randint(200, (n, 2))), torch.randn(n))
        ds = DatasetTuple(*a)
        with pytest.raises(RuntimeError) as ex:
            len(ds)
        assert str(ex.value) == "Need all tensors to have same lenght."

    def test_getitem(self):
        torch.manual_seed(123)
        n = 10
        a = ((torch.randn(n, 3), torch.randint(200, (n, 2))), torch.randn(n))
        ds = DatasetTuple(*a)
        assert_tupletree_equal(ds[0], ds[[0]])
        assert_tupletree_equal(ds[0], ds[:1])
        assert_tupletree_equal(ds[2:5], ds[[2, 3, 4]])

class TestDataLoaderSlice:
    @pytest.mark.parametrize('batch_size', [3, 10])
    @pytest.mark.parametrize('num_workers', [0, 2])
    def test_next_iter(self, batch_size, num_workers):
        torch.manual_seed(123)
        n = 20
        a = ((torch.randn(n, 3), torch.randint(200, (n, 2))), torch.randn(n))
        a = tuplefy(a)
        ds = DatasetTuple(*a)
        dl = DataLoaderSlice(ds, batch_size, False, num_workers=num_workers)
        a = a.iloc[:batch_size]
        b = next(iter(dl))
        assert_tupletree_equal(a, b)

