# -*- coding: utf-8 -*-

"""Top-level package for pyth."""

__author__ = """Haavard Kvamme"""
__email__ = 'haavard.kvamme@gmail.com'
__version__ = '0.0.0'

from . import base, callbacks, fitnet, practical, optim
# from . import callbacks
# from . import fitnet

from .base import Model
from .data import DatasetTuple, DataLoaderSlice
