import functools
import itertools
import operator
import numpy as np
import torch
from pyth.data import DatasetTuple, DataLoaderSlice


class Tuple(tuple):
    """Planning to extend this"""
    def apply(self, func):
        return apply_tuple(func)(self)

    def reduce(self, func, init_func=None, **kwargs):
        return reduce_tuple(func, init_func)(self, **kwargs)
    
    def shapes(self):
        return shapes_of(self)
    
    def lens(self):
        return lens_of(self)
    
    def dtypes(self):
        return dtypes_of(self)
    
    def to_tensor(self):
        if type(self) is torch.Tensor:
            return self
        return numpy_to_tensor(self)

    def to_numpy(self):
        if type(self) is np.ndarray:
            return self
        return tensor_to_numpy(self)
    
    def type(self):
        return type_of(self)
    
    def types(self):
        return types_of(self)

    def astype(self, dtype, *args, **kwargs):
        return astype(self, dtype, *args, **kwargs)

    def is_flat(self):
        return is_flat(self)

    def flatten(self):
        return flatten_tuple(self)

    def to_levels(self):
        return tuple_levels(self)

    def cat(self, dim=0):
        return cat(self, dim=0)

    def reduce_nrec(self, func):
        """Reduct non-recursive, only first list."""
        return functools.reduce(func, self)

    def apply_nrec(self, func):
        """Apply non-recursive, only first list"""
        return Tuple(func(sub) for sub in self)
    
    def all(self):
        if not self.is_flat():
            raise RuntimeError("Need to have a flat structure to use 'all'")
        return all(self)

    def tuplefy(self, types=(list, tuple)):
        self = tuplefy(self, types=types, stop_at_tuple=False)
        return self

    def split(self, split_size, dim=0):
        return split(self, split_size, dim)

    def agg_list(self):
        """Aggregate data to a list of the data
        ((a1, (a2, a3)), (b1, (b2, b3))) -> ([a1, b1], ([a2, b2], [a3, b3]))

        Inverse of split_agg
        """
        return agg_list(self)

    def split_agg(self):
        """The inverse opeation of agg_list
        ([a1, b1], ([a2, b2], [a3, b3])) -> ((a1, (a2, a3)), (b1, (b2, b3)))
        """
        return split_agg(self)

    def all_equal(self):
        """All typles (from top level) are the same
        E.g. (a, a, a)
        """
        return all_equal(self)

    def to_device(self, device):
        return to_device(self, device)

    def make_dataloader(self, batch_size, shuffle, num_workers=0):
        return make_dataloader(self, batch_size, shuffle, num_workers)

    @property
    def iloc(self):
        """Used in as pd.DataFrame.iloc for subsetting tensors and arrays"""
        return _TupleSlicer(self)


class _TupleSlicer:
    def __init__(self, tuple_):
        self.tuple_ = tuple_

    def __getitem__(self, index):
        return self.tuple_.apply(lambda x: x[index])

# _CONTAINERS = (list, tuple, Tuple)
_CONTAINERS = (Tuple,)

def apply_tuple(func):
    """Apply a function to data in tuples of list

    E.g.: Two ways to get shapes of all elements in a tuple

    data = [(torch.randn(1, 2), torch.randn(2, 3)),
            torch.randn(4, 4)]

    # Method 1:
    @apply_tuple
    def shape_of(data):
        return len(data)
    
    shape_of(data)

    # Method 2:
    apply_tuple(lambda x: x.shape)(data)
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if type(data) in _CONTAINERS:
            return Tuple(wrapper(sub, *args, **kwargs) for sub in data)
        return func(data, *args, **kwargs)
    return wrapper

def reduce_tuple(func, init_func=None):
    """Reduce opration on tuples.
    Works recursively on objects that are not Tuple:

    Exs:
    a = ((1, (2, 3), 4),
         (1, (2, 3), 4),
         (1, (2, 3), 4),)
    a = tuplefy(a)
    reduce_tuple(lambda x, y: x+y)(a)

    Gives:
    (3, (6, 9), 12)
    """
    def reduce_rec(acc_val, val, **kwargs):
        if type(acc_val) in _CONTAINERS:
            return Tuple(reduce_rec(av, v) for av, v in zip(acc_val, val))
        return func(acc_val, val, **kwargs)

    @functools.wraps(func)
    def wrapper(data, **kwargs):
        if not data.to_levels().all_equal():
        # if not data.to_levels().reduce_nrec(operator.eq):
            raise ValueError("Topology is not the same for all elements in data, and can not be reduced")
        iterable = iter(data)
        if init_func is None:
            acc_val = next(iterable)
        else:
            acc_val = data[0].apply(init_func)
        for val in iterable:
            acc_val = reduce_rec(acc_val, val, **kwargs)
        return acc_val
    return wrapper

def all_equal(data):
    """All typles (from top level) are the same
    E.g. (a, a, a)
    """
    return data.apply_nrec(lambda x: x == data[0]).all()

def agg_list(data):
    """Aggregate data to a list of the data
    ((a1, (a2, a3)), (b1, (b2, b3))) -> ([a1, b1], ([a2, b2], [a3, b3]))

    Inverse of split_agg
    """
    init_func = lambda _: list()
    def append_func(list_, val):
        list_.append(val)
        return list_
    return reduce_tuple(append_func, init_func)(data)

@apply_tuple
def shapes_of(data):
    """Apply x.shape to elemnts in data."""
    return data.shape

@apply_tuple
def lens_of(data):
    """Apply len(x) to elemnts in data."""
    return len(data)

@apply_tuple
def dtypes_of(data):
    """Apply x.dtype to elemnts in data."""
    return data.dtype

@apply_tuple
def numpy_to_tensor(data):
    """Transform numpy arrays to torch tensors."""
    return torch.from_numpy(data)

@apply_tuple
def tensor_to_numpy(data):
    """Transform torch tensort arrays to numpy arrays."""
    if hasattr(data, 'detach'):
        data = data.detach()
    if type(data) is torch.Size:
        return np.array(data)
    return data.numpy()

@apply_tuple
def astype(data, dtype, *args, **kwargs):
    """Change type to dtype.

    torch tensors: we call 'data.type(dtype, *args, **kwargs)'
    numpy arrays: we call 'data.astype(dtype, *args, **kwargs)'
    """
    if type(data) is torch.Tensor:
        return data.type(dtype, *args, **kwargs)
    elif type(data) is np.ndarray:
        return data.astype(dtype, *args, **kwargs)
    else:
        return RuntimeError(
            f"""Need 'data' to be torch.tensor of np.ndarray, found {type(data)}.
            """)

@apply_tuple
def types_of(data):
    """Returns all types in data"""
    return type(data)

def type_of(data):
    """Returns THE type of subelements in data.
    Hence, all elements in data needs to have the same class.
    """
    types = data.types().flatten()
    if types.count(types[0]) != len(types):
        raise ValueError("All objects in 'data' doest have the same type.")
    return types[0]

def is_flat(data):
    if type(data) not in _CONTAINERS:
        return True
    return all(data.apply_nrec(lambda x: type(x) not in _CONTAINERS))

def flatten_tuple(data):
    """Flatten tuple"""
    if type(data) not in _CONTAINERS:
        return data
    new = Tuple(sub if type(sub) in _CONTAINERS else (sub,) for sub in data)
    new = Tuple(itertools.chain.from_iterable(new))
    if new.is_flat():
        return new
    return flatten_tuple(new)

def tuple_levels(data, level=-1):
    """Replaces objects with the level they are on.
    
    Arguments:
        data {list or tuple} -- Data
    
    Keyword Arguments:
        level {int} -- Start level. Default of -1 gives flat list levels 0 (default: {-1})
    
    Returns:
        tuple -- Levels of objects
    """
    if type(data) not in _CONTAINERS:
        return level
    return Tuple(tuple_levels(sub, level+1) for sub in data)

def cat(seq, dim=0):
    """Conatenate tensors/arrays in tuple.
    Only works for dim=0, meaning we concatenate in the batch dim.
    """
    if dim != 0:
        raise NotImplementedError
    if not seq.shapes().apply(lambda x: x[1:]).all_equal():
        raise ValueError("Shapes of merged arrays need to be the same")

    type_ = seq.type()
    agg = seq.agg_list()
    if type_ is torch.Tensor:
        return agg.apply(torch.cat)
    elif type_ is np.ndarray:
        return agg.apply(np.concatenate)
    raise RuntimeError(f"Need type to be np.ndarray or torch.Tensor, fournd {type_}.")

def split(data, split_size, dim=0):
    """Use torch.split"""
    if dim != 0:
        raise NotImplementedError
    if data.type() is not torch.Tensor:
        raise NotImplementedError("Only implemented for torch tensors because np.split works differently")

    splitted = data.apply(lambda x: x.split(split_size))
    return split_agg(splitted)

def split_agg(agg):
    """The inverse opeation of agg_list"""
    if type(agg) is Tuple:
        new = agg.apply_nrec(split_agg)
        return Tuple(zip(*new)).tuplefy()
    return agg

# def tuplefy(data, types=(list, tuple)):
#     """Make Tuple object by recursively changing all 'types' to Tuple
#     Use 'Tuple(data)' to only change top level.
#     Use instead 'tuplefy_if_not(data)' if you do not want to tuplefy existing
#     Tuple objects.
#     """
#     def _tuplefy(data, types):
#         if type(data) in types:
#             return Tuple(_tuplefy(sub, types) for sub in data)
#         return data
#     types = list(types)
#     types.append(Tuple)
#     if type(data) not in types:
#         return Tuple((data,))
#     return _tuplefy(data, types)
# def tuplefy(*data, types=(list, tuple)):
#     """Make Tuple object by recursively changing all 'types' to Tuple
#     Use 'Tuple(data)' to only change top level.
#     Use instead 'tuplefy_if_not(data)' if you do not want to tuplefy existing
#     Tuple objects.
#     """
#     def _tuplefy(data, types):
#         if type(data) in types:
#             return Tuple(_tuplefy(sub, types) for sub in data)
#         return data
#     types = list(types) + [Tuple]
#     # types.append(Tuple)
#     if (len(data) == 1) and (type(data[0]) in types):
#         data = data[0]
#     data = Tuple(data)
#     return _tuplefy(data, types)
def tuplefy(*data, types=(list, tuple), stop_at_tuple=True):
    """Make Tuple object from *args.
    
    Keyword Arguments:
        types {tuple} -- Types that should be transformed to Tuple (default: {(list, tuple)})
        stop_at_tuple {bool} -- If 'True', the recusion stops at Tuple elements,
            and if 'False' it will continue through Tuple elements. (default: {True})
    
    Returns:
        Tuple -- A Tuple object
    """
    def _tuplefy(data, first=False):
        if (type(data) in types) or first:
            return Tuple(_tuplefy(sub) for sub in data)
        return data

    types = list(types)
    if not stop_at_tuple:
        types.extend(_CONTAINERS)
    if (len(data) == 1) and ((type(data[0]) in types) or (type(data[0]) in _CONTAINERS)):
        data = data[0]
    data = Tuple(data)
    return _tuplefy(data, first=True)


# def tuplefy(*data, types=(list, tuple), stop_at_tuple=True):
#     """Make Tuple object by recursively changing all 'types' to Tuple
#     Use 'Tuple(data)' to only change top level.
#     Use instead 'tuplefy_if_not(data)' if you do not want to tuplefy existing
#     Tuple objects.
#     """
#     if stop_at_tuple:
#         return tuplefy_if_not(*data, types=types)

#     def _tuplefy(data, types):
#         if type(data) in types:
#             return Tuple(_tuplefy(sub, types) for sub in data)
#         return data
#     types = list(types) + [Tuple]
#     # types.append(Tuple)
#     if (len(data) == 1) and (type(data[0]) in types):
#         data = data[0]
#     data = Tuple(data)
#     return _tuplefy(data, types)

# def tuplefy_if_not(*data, types=(list, tuple)):
#     """Call tuplefy on data if data is not Tuple (recursively making all Tuple).
#     If data is Tuple, leave strucure unchanged.
#     """
#     def _func(x):
#         if type(x) is Tuple:
#             return x
#         if type(x) not in types:
#             return x
#         return tuplefy(x, types=types, stop_at_tuple=False)

#     data = Tuple(data).apply_nrec(_func)
#     if len(data) == 1:
#         return _tuple_if_not(data[0], types)
#     return data

# def _tuple_if_not(data, types=(list, tuple)):
#     if type(data) not in (list(types) + [Tuple]):
#         data = (data,)
#     return Tuple(data)

@apply_tuple
def to_device(data, device):
    if type(data) is not torch.Tensor:
        raise RuntimeError(f"Need 'data' to be tensors, not {type(data)}.")
    return data.to(device)

def make_dataloader(data, batch_size, shuffle, num_workers=0):
    dataset = DatasetTuple(data)
    dataloader = DataLoaderSlice(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


## Some tests that should be written
# a = pyth.Tuple((1, (2, (3, 4))))

# assert a == a.tuplefy()
# assert a == tuplefy(a)

# b = [1, [2, [3, 4]]]

# assert a == tuplefy(b)
# assert a == tuplefy(b[0], b[1])
# assert tuplefy(1) == (1,)