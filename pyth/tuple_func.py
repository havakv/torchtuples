import functools
import itertools
import operator
import numpy as np
import torch


class ReductionList(list):
    """Identical to list, but in not considered a 
    list or tuple by the functioal api.
    """
    pass

class Tuple(tuple):
    """Planning to extend this"""
    def apply(self, func):
        return apply_tuple(func)(self)
    
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

    def to_tuple(self, types=(list, tuple)):
        self = to_tuple(self, types)
        return self

    def split(self, split_size, dim=0):
        return split(self, split_size, dim)



# planning to remove list and tuple from this
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

    
# @apply_tuple
# def classes_of(data):
#     """Returns all calsses in data"""
#     return type(data)

# def class_of(data):
#     """Returns THE class of subelements in data.
#     Hence, all elements in data needs to have the same class.
#     """
#     classes = classes_of(data)
#     classes = flatten_tuple(classes)
#     if classes.count(classes[0]) != len(classes):
#         raise ValueError("All objects in 'data' doest have the same class.")
#     return classes[0]

# type_of = class_of
# types_of = classes_of

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

def append_list(data, new):
    if type(data) not in _CONTAINERS:
        assert type(new) is ReductionList, "Need lists in 'new' to be RecutionList"
        new.append(data)
        return new
    for d, n in zip(data, new):
        append_list(d, n)
    return new

def agg_list(data):
    new = apply_tuple(lambda _: ReductionList())(data[0])
    for sub in data:
        append_list(sub, new)
    return new

def cat(seq, dim=0):
    """Conatenate tensors/arrays in tuple.
    Only works for dim=0, meaning we concatenate in the batch dim.
    """
    if dim != 0:
        raise NotImplementedError
        # Would need to fix shapes for this!!!!
    if not seq.to_levels().reduce_nrec(operator.eq):
        raise ValueError("Topology is not the same for all elements in seq")
    if not seq.shapes().apply(lambda x: x[1:]).reduce_nrec(operator.eq):
        raise ValueError("Shapes of merged arrays need to be the same")

    type_ = seq.type()
    agg = agg_list(seq)
    if type_ is torch.Tensor:
        return agg.apply(torch.cat)
    elif type_ is np.ndarray:
        return agg.apply(np.concatenate)
    raise RuntimeError(f"Need type to be np.ndarray or torch.Tensor, fournd {type_}.")

def split(data, split_size, dim=0):
    if dim != 0:
        raise NotImplementedError
    if data.type() is not torch.Tensor:
        raise NotImplementedError("Only implemented for torch tensors because np.split works differently")

    splitted = data.apply(lambda x: x.split(split_size))
    return split_agg(splitted)

def split_agg(agg):
    if type(agg) is Tuple:
        new = agg.apply_nrec(split_agg)
        return Tuple(zip(*new)).to_tuple()
    # elif type(agg) is ReductionList:
    return agg

def to_tuple(data, types=(list, tuple)):
    types = list(types)
    types.append(Tuple)
    if type(data) in types:
        return Tuple(to_tuple(sub) for sub in data)
    return data