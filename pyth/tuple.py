import functools
import itertools
import operator
import numpy as np
import torch
from pyth.data import DatasetTuple, DataLoaderSlice


def apply_leaf(func):
    """Apply a function to data in Tuple objects (leaf nodes).

    E.g.: Two ways to get shapes of all elements in a tuple

    data = tuplefy([(torch.randn(1, 2), torch.randn(2, 3)),
                    torch.randn(4, 4)])

    # Method 1:
    @apply_leaf
    def shape_of(data):
        return len(data)
    
    shape_of(data)

    # Method 2:
    apply_leaf(lambda x: x.shape)(data)
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if type(data) in _CONTAINERS:
            return Tuple(wrapper(sub, *args, **kwargs) for sub in data)
        return func(data, *args, **kwargs)
    return wrapper

def reduce_leaf(func, init_func=None):
    """Reduce operation on Tuple objects.
    It reduces the leaf nodes to the first Tuple's topology.

    Exs:
    a = ((1, (2, 3), 4),
         (1, (2, 3), 4),
         (1, (2, 3), 4),)
    a = tuplefy(a)
    reduce_leaf(lambda x, y: x+y)(a)

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

def zip_leaf(data):
    """Aggregate data to a list of the data.
    This is essentialy a zip opperation that works on the leaf nodes
    ((a1, (a2, a3)), (b1, (b2, b3))) -> ([a1, b1], ([a2, b2], [a3, b3]))

    Inverse of unzip_leaf
    """
    init_func = lambda _: list()
    def append_func(list_, val):
        list_.append(val)
        return list_
    return reduce_leaf(append_func, init_func)(data)

@apply_leaf
def shapes_of(data):
    """Apply x.shape to elemnts in data."""
    return data.shape

@apply_leaf
def lens_of(data):
    """Apply len(x) to elemnts in data."""
    return len(data)

@apply_leaf
def dtypes_of(data):
    """Apply x.dtype to elemnts in data."""
    return data.dtype

@apply_leaf
def numpy_to_tensor(data):
    """Transform numpy arrays to torch tensors."""
    return torch.from_numpy(data)

@apply_leaf
def tensor_to_numpy(data):
    """Transform torch tensort arrays to numpy arrays."""
    if hasattr(data, 'detach'):
        data = data.detach()
    if type(data) is torch.Size:
        return np.array(data)
    return data.numpy()

@apply_leaf
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

@apply_leaf
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
    """Returns true if the Tuple data is flat"""
    if type(data) not in _CONTAINERS:
        return True
    return all(data.apply_nrec(lambda x: type(x) not in _CONTAINERS))

def flatten_tuple(data):
    """Flatten the Tuple data"""
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
    agg = seq.zip_leaf()
    if type_ is torch.Tensor:
        return agg.apply(torch.cat)
    elif type_ is np.ndarray:
        return agg.apply(np.concatenate)
    raise RuntimeError(f"Need type to be np.ndarray or torch.Tensor, fournd {type_}.")

def split(data, split_size, dim=0):
    """Use torch.split and create multiple Tuples with the splitted tensors."""
    if dim != 0:
        raise NotImplementedError
    if data.type() is not torch.Tensor:
        raise NotImplementedError("Only implemented for torch tensors because np.split works differently")

    splitted = data.apply(lambda x: x.split(split_size))
    return unzip_leaf(splitted)

def unzip_leaf(agg):
    """The inverse opeation of zip_leaf.
    This is essentialy a zip(*agg) opperation that works on the leaf nodes
    ([a1, b1], ([a2, b2], [a3, b3])) -> ((a1, (a2, a3)), (b1, (b2, b3))) 
    """
    if type(agg) in _CONTAINERS:
        new = agg.apply_nrec(unzip_leaf)
        return Tuple(zip(*new)).tuplefy()
    return agg

def tuplefy(*data, types=(list, tuple), stop_at_tuple=True):
    """Make Tuple object from *args.
    
    Keyword Arguments:
        types {tuple} -- Types that should be transformed to Tuple (default: {(list, tuple)})
        stop_at_tuple {bool} -- If 'True', the recusion stops at Tuple elements,
            and if 'False' it will continue through Tuple elements. 
    
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

@apply_leaf
def to_device(data, device):
    """Move data to device
    
    Arguments:
        data {Tuple, tensor} -- Tensors that should be moved to device.
        device {str, torch.device} -- Device data is moved to.
    
    Returns:
        Tuple, tensor -- Data moved to device
    """
    if type(data) is not torch.Tensor:
        raise RuntimeError(f"Need 'data' to be tensors, not {type(data)}.")
    return data.to(device)

def make_dataloader(data, batch_size, shuffle, num_workers=0, to_tensor=True):
    """Create a dataloder from tensor or np.arrays.
   
    Arguments:
        data {tuple, np.array, tensor} -- Data in dataloader e.g. (x, y)
        batch_size {int} -- Batch size used in dataloader
        shuffle {bool} -- If order should be suffled
    
    Keyword Arguments:
        num_workers {int} -- Number of workers in dataloader (default: {0})
        to_tensor {bool} -- Ensure that we use tensors (default: {True})
    
    Returns:
        DataLoaderSlice -- A dataloader object like the torch DataLoader
    """
    if to_tensor:
        data = tuplefy(data).to_tensor()
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

def docstring(doc_func):
    """Decorator to make function have the docstring of 'doc_func'."""
    def docstring_real(func):
        func.__doc__ = doc_func.__doc__
        return func
    return docstring_real


class Tuple(tuple):
    """A tuple with some methods that works recursively.
    This is essentially a tree structure, with all internal nodes being Tuple objects,
    and all leaf nodes contain data.

    Hence the apply methods is a map function on the leaf nodes.
    """
    def apply(self, func):
        """Shorthand to apply_leaf(func)(self)"""
        return apply_leaf(func)(self)

    def reduce(self, func, init_func=None, **kwargs):
        """Shorthand to reduce_leaf(func, init_func)(self).
        It reduces the leaf nodes to the first Tuple's topology.

        Exs:
        a = ((1, (2, 3), 4),
             (1, (2, 3), 4),
             (1, (2, 3), 4),)
        Tuple(a).reduce(lambda x, y: x+y) 

        Gives: (3, (6, 9), 12)
        """
        return reduce_leaf(func, init_func)(self, **kwargs)
    
    @docstring(shapes_of)
    def shapes(self):
        return shapes_of(self)
    
    @docstring(lens_of)
    def lens(self):
        return lens_of(self)
    
    @docstring(dtypes_of)
    def dtypes(self):
        return dtypes_of(self)
    
    @docstring(numpy_to_tensor)
    def to_tensor(self):
        if self.type() is torch.Tensor:
            return self
        return numpy_to_tensor(self)

    @docstring(tensor_to_numpy)
    def to_numpy(self):
        if self.type() is np.ndarray:
            return self
        return tensor_to_numpy(self)
    
    @docstring(type_of)
    def type(self):
        return type_of(self)
    
    @docstring(types_of)
    def types(self):
        return types_of(self)

    @docstring(astype)
    def astype(self, dtype, *args, **kwargs):
        return astype(self, dtype, *args, **kwargs)

    @docstring(is_flat)
    def is_flat(self):
        return is_flat(self)

    @docstring(flatten_tuple)
    def flatten(self):
        return flatten_tuple(self)

    @docstring(tuple_levels)
    def to_levels(self):
        return tuple_levels(self)

    @docstring(cat)
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

    @docstring(tuplefy)
    def tuplefy(self, types=(list, tuple), stop_at_tuple=False):
        self = tuplefy(self, types=types, stop_at_tuple=stop_at_tuple)
        return self

    @docstring(split)
    def split(self, split_size, dim=0):
        return split(self, split_size, dim)

    @docstring(zip_leaf)
    def zip_leaf(self):
        return zip_leaf(self)

    @docstring(unzip_leaf)
    def unzip_leaf(self):
        return unzip_leaf(self)

    @docstring(all_equal)
    def all_equal(self):
        """All typles (from top level) are the same
        E.g. (a, a, a)
        """
        return all_equal(self)

    @docstring(to_device)
    def to_device(self, device):
        return to_device(self, device)

    @docstring(make_dataloader)
    def make_dataloader(self, batch_size, shuffle, num_workers=0):
        return make_dataloader(self, batch_size, shuffle, num_workers)

    @property
    def iloc(self):
        """Used in as pd.DataFrame.iloc for subsetting tensors and arrays"""
        return _TupleSlicer(self)

    def pipe(self, func, *arg, **kwargs):
        """Shorthand for func(self, *args, **kwargs)"""
        return func(self, *args, **kwargs)


class _TupleSlicer:
    def __init__(self, tuple_):
        self.tuple_ = tuple_

    def __getitem__(self, index):
        return self.tuple_.apply(lambda x: x[index])


# _CONTAINERS = (list, tuple, Tuple)
_CONTAINERS = (Tuple,)