import pytest
import numpy as np
import torch
import torchtuples as tt
from torchtuples.utils import array_or_tensor

def make_data(multi_in=False, multi_out=False, cl=tuple, dl=False, data_torch=False):
    input = torch.randn(5)
    if multi_in:
        input = (input, (input, input))
        input = cl(input)
    tensor = torch.randn(5)
    if multi_out:
        tensor = ((tensor, tensor), tensor)
        tensor = cl(tensor)
    if data_torch is False:
        input, tensor = tt.tuplefy(input, tensor).to_numpy()
    if dl is True:
        input = tt.tuplefy(input).make_dataloader(10, False)
    return input, tensor

@pytest.mark.parametrize('numpy', (True, False))
@pytest.mark.parametrize('multi_in', (True, False))
@pytest.mark.parametrize('multi_out', (True, False))
@pytest.mark.parametrize('cl', (tuple, list, tt.tuplefy))
@pytest.mark.parametrize('dl', (True, False))
@pytest.mark.parametrize('data_torch', (True, False))
def test_array_or_tensor_type_numpy(numpy, multi_in, multi_out, cl, dl, data_torch):
    input, tensor = make_data(multi_in, multi_out, cl, dl, data_torch)
    out = array_or_tensor(tensor, numpy, input)
    if multi_out is True:
        assert type(out) is tt.TupleTree
    else:
        assert type(out) in [np.ndarray, torch.Tensor]
    assert numpy is (tt.tuplefy(out).type() is np.ndarray)


@pytest.mark.parametrize('multi_in', (True, False))
@pytest.mark.parametrize('multi_out', (True, False))
@pytest.mark.parametrize('cl', (tuple, list, tt.tuplefy))
@pytest.mark.parametrize('dl', (True, False))
@pytest.mark.parametrize('data_torch', (True, False))
def test_array_or_tensor_type_none(multi_in, multi_out, cl, dl, data_torch):
    numpy = None
    input, tensor = make_data(multi_in, multi_out, cl, dl, data_torch)
    out = array_or_tensor(tensor, numpy, input)
    if multi_out is True:
        assert type(out) is tt.TupleTree
    else:
        assert type(out) in [np.ndarray, torch.Tensor]
    correct_type = np.ndarray if (dl is True) or (data_torch is False) else torch.Tensor
    assert tt.tuplefy(out).type() is correct_type
