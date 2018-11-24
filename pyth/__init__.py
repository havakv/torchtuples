# -*- coding: utf-8 -*-

"""Top-level package for pyth."""

__author__ = """Haavard Kvamme"""
__email__ = 'haavard.kvamme@gmail.com'
__version__ = '0.0.0'

# from . import base, callbacks, fitnet, practical, optim
import pyth.base
import pyth.callbacks
import pyth.practical
import pyth.optim
import pyth.tuple
# from . import callbacks
# from . import fitnet

from pyth.base import Model
from pyth.data import DatasetTuple, DataLoaderSlice

# from pyth.base import class_of, numpy_to_tensor, tensor_to_dataloader, numpy_to_dataloader,\
#     to_device, tuple_if_tensor

from pyth.tuple import Tuple, tuplefy, to_device, numpy_to_tensor, tensor_to_numpy

