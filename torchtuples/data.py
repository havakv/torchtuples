'''
Small changes to the pytorch dataloader. 
Now works with with batches (slicing rather than loop over indexes), 
and can potentially stop workers from shutting down at each batch.
'''
# import random
import warnings
import copy
import torch
# from torch.utils.data.dataloader import DataLoader, RandomSampler 
# from torch.utils.data import Dataset
import torch.utils.data

import torchtuples
from torchtuples._pytorch_dataloader import _DataLoaderIterSlice


class DataLoaderSlice(torch.utils.data.dataloader.DataLoader):
    r"""
    Like DataLoader but works on batches instead of iterating
    through the batch.

    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
            If None, we simply pass the the in put to collate_fn.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=None, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        if collate_fn is None:
            collate_fn = DataLoaderSlice._identity
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                         collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

    def __iter__(self):
        return _DataLoaderIterSlice(self)

    @staticmethod
    def _identity(x):
        '''Function returning x'''
        return x


class RandomSamplerContinuous(torch.utils.data.dataloader.RandomSampler):
    """Samples elements randomly, without replacement, and continues for ever.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __iter__(self):
        while True:
            for i in iter(torch.randperm(len(self.data_source)).long()):
                yield i


class DatasetTuple(torch.utils.data.Dataset):
    """Dataset where input and target can be tuples.

    Arguments:
        input {tuple or list} -- What is passed to the network (list of x tensors.)
        target {tuple or list} -- Label information passed to the loss function
            (list of y tensors)
    """
    def __init__(self, *data):
        self.data = torchtuples.tuplefy(*data)
        if not self.data.apply(lambda x: type(x) is torch.Tensor).flatten().all():
            warnings.warn("All data is not torch.Tensor. Consider fixing this.")

    def __getitem__(self, index):
        if (not hasattr(index, '__iter__')) and (type(index) is not slice):
            index = [index]
        return self.data.iloc[index]

    def __len__(self):
        lens = self.data.lens().flatten()
        if not lens.all_equal():
            raise RuntimeError("Need all tensors to have same lenght.")
        return lens[0]


class DatasetInputOnly:
    """Class for chaning a Dataset contraining inputs and targets
    to only return the inputs.
    Usefurll for predict methods.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0]


def dataloader_input_only(dataloader):
    """Create a new dataloader the only returns the inputs and not the targets.
    Useful when calling predict:

        dataloader = ...
        model.fit(dataloader)
        dl_input = dataloader_input_only(dataloader)
        model.predict(dl_inpt)

    See e.g. MNIST examples code.
    """
    if type(dataloader.sampler) is not torch.utils.data.sampler.SequentialSampler:
        warnings.warn("Dataloader might not be deterministic!")
    dl_new = copy.copy(dataloader)
    dl_new.dataset = DatasetInputOnly(dl_new.dataset)
    return dl_new
