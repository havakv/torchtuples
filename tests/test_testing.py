import pytest
import numpy as np
import torch
from torchtuples.tupletree import tuplefy
from torchtuples.testing import assert_tupletree_equal

@pytest.mark.parametrize('a, b, ex_string', [
    ((1, 2), (1,), 'Not TupleTree'),
    (tuplefy(1, 2), tuplefy(1, (2,)), 'Not same structure'),
    (tuplefy(np.arange(5)), tuplefy(torch.arange(5)), 'Not same types'),
    (tuplefy(np.arange(5)), tuplefy(np.arange(5).astype('float')), 'Not same dtype'),
    (tuplefy(torch.arange(5)), tuplefy(torch.arange(5).float()), 'Not same dtype'),
    (tuplefy(np.arange(5)), tuplefy(np.arange(1, 6)), 'Not equal values'),
    (tuplefy(torch.arange(5)), tuplefy(torch.arange(1, 6)), 'Not equal values'),
])
def test_assert_tupletree_equal_fails(a, b, ex_string):
    with pytest.raises(AssertionError) as ex:
        assert_tupletree_equal(a, b)
    assert str(ex.value) == ex_string

@pytest.mark.parametrize('a, b, check_dtypes', [
    (tuplefy(1, 2), tuplefy(1, 2), True),
    (tuplefy(np.arange(5), (1, 2)), tuplefy(np.arange(5).astype('float'), (1, 2)), False),
])
def test_assert_tupletree_equal_pass(a, b, check_dtypes):
    assert_tupletree_equal(a, b, check_dtypes)
