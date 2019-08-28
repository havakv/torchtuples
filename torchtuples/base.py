
"""Model for fitting torch models.
"""
import os
import time
import random
from collections import OrderedDict, defaultdict
import warnings
import contextlib
import numpy as np  # shoul be remved
import torch
import torchtuples.callbacks as cb
from torchtuples.optim import AdamW, OptimWrap
from torchtuples.tupletree import tuplefy, TupleTree, make_dataloader
from torchtuples.utils import make_name_hash


class Model(object):
    """Train torch models using dataloaders, tensors or np.arrays.
    
    Arguments:
        net {torch.nn.Module} -- A torch module.
    
    Keyword Arguments:
        loss {function} -- Set function that is used for training 
            (e.g. binary_cross_entropy for torch) (default: {None})
        optimizer {Optimizer} -- A torch optimizer or similar. Preferrably use torchtuples.optim instead of
            torch.optim, as this allows for reinitialization, etc. If 'None' set to torchtuples.optim.AdamW.
            (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferrably pass a torch.device object.
            If 'None': use default gpu if avaiable, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    Example simple model:
    ---------------------
    from torchtuples import Model
    import torch
    from torch import nn
    from torch.nn import functional as F

    n_rows, n_covs = 1000, 4
    x = torch.randn((n_rows, n_covs))
    y = 2 * x.sum(1, keepdim=True) + 4  # y = 2 * x + 4

    net = nn.Sequential(nn.Linear(n_covs, 10), nn.ReLU(), nn.Linear(10, 1))
    loss = F.mse_loss
    model = Model(net, loss)
    log = model.fit(x, y, batch_size=32, epochs=30)
    log.plot()
    """
    def __init__(self, net, loss=None, optimizer=None, device=None):
        self.net = net
        if type(self.net) is str:
            self.load_net(self.net)
        self.loss = loss
        self.optimizer = optimizer if optimizer is not None else AdamW
        self.device = self._device_from__init__(device)
        self.net.to(self.device)
        # self.net_predict = net_predict if net_predict else self.net
        # self.net_predict.to(self.device)
        if not hasattr(self, 'make_dataloader_predict'):
            self.make_dataloader_predict = self.make_dataloader
        self._init_train_log()

    def _init_train_log(self):
        self.log = cb.TrainingLogger()
        self.train_metrics = cb._MonitorFitMetricsTrainData()
        self.val_metrics = cb.MonitorFitMetrics()
        self.log.monitors = OrderedDict(train_=self.train_metrics, val_=self.val_metrics)
        self.callbacks = None
    
    @staticmethod
    def _device_from__init__(device):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif type(device) is str:
            device = torch.device(device)
        elif type(device) is int:
            device = torch.device(f"cuda:{device}")
        if type(device) is not torch.device:
            raise ValueError("Argument 'device' needs to be None, string, int or torch.device object.")
        return device
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if callable(self._optimizer):
            self._optimizer = self._optimizer(params=self.net.parameters())
        if not isinstance(self._optimizer, OptimWrap):
            self._optimizer = OptimWrap(self._optimizer)

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0, **kwargs):
        """Function for creating a dataloader from tensors or arrays.
        It is natural to rewrite this method in inherited classes.

        self.make_dataloader_predict will be set to this method if not implemented
        separatelly.

        This simply calls tupletree.make_dataloader, but is included to make
        inheritance simpler. 

        Arguments:
            data {tuple, np.array, tensor} -- Data in dataloader e.g. (x, y)
            batch_size {int} -- Batch size used in dataloader
            shuffle {bool} -- If order should be suffled

        Keyword Arguments:
            num_workers {int} -- Number of workers in dataloader (default: {0})
            to_tensor {bool} -- Ensure that we use tensors (default: {True})
            **kwargs -- Passed to make_dataloader.

        Returns:
            DataLoaderSlice -- A dataloader object like the torch DataLoader
        """
        dataloader = make_dataloader(data, batch_size, shuffle, num_workers, **kwargs)
        return dataloader
    
    def _setup_train_info(self, dataloader):
        self.fit_info = {'batches_per_epoch': len(dataloader)}
        def _tuple_info(tuple_):
            tuple_ = tuplefy(tuple_)
            return {'levels': tuple_.to_levels(), 'shapes': tuple_.shapes().apply(lambda x: x[1:])}
        # input, target = dataloader.dataset[0]
        data = _get_element_in_dataloader(dataloader)
        if data is not None:
            input, target = data
            self.fit_info['input'] = _tuple_info(input)
    
    def _to_device(self, data):
        """Move data to self.device.
        
        Arguments:
            data {tensor or tuple} -- Data
        
        Returns:
            tensor or tuple -- Data moved to device.
        """
        if data is None:
            return tuplefy(data)
        return tuplefy(data).to_device(self.device)

    def compute_metrics(self, input, target, metrics):
        """Function for computing loss.
        Is rather general, but can be reimpliemented by sub classes.
        
        Arguments:
            input {tensor or tuple} -- This should be passed to self.net.
            target {tensor or tuple} -- This is the targets that should be used in self.loss.
            metrics (dictionary with funcs) -- Metrics that should be computed.

        Returns:
            tensor -- Results of self.loss()
        """
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        input = self._to_device(input)
        target = self._to_device(target)
        out = self.net(*input)
        out = tuplefy(out)
        res = {name: metric(*out, *target) for name, metric in metrics.items()}
        return res

    def _setup_metrics(self, metrics):
        all_metrics = {'loss': self.loss}
        if metrics is not None:
            if not hasattr(metrics, 'items'):
                if not hasattr(metrics, '__iter__'):
                    metrics = [metrics]
                metrics = {met.__name__: met for met in metrics}
            if 'loss' in metrics:
                raise ValueError("The 'loss' keyword is reserved for the loss function.")
            all_metrics.update(metrics)
        return all_metrics

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None,
                       val_dataloader=None):
        """Fit a dataloader object.
        See 'fit' for tensors and np.arrays.
        
        Arguments:
            dataloader {dataloader} -- A dataloader that gives (input, target).
        
        Keyword Arguments:
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
        
        Returns:
            TrainingLogger -- Training log
        """
        self._setup_train_info(dataloader)
        self.metrics = self._setup_metrics(metrics)
        self.log.verbose = verbose
        self.val_metrics.dataloader = val_dataloader
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.TrainingCallbackHandler(self.optimizer, self.train_metrics, self.log,
                                                    self.val_metrics, callbacks)
        self.callbacks.give_model(self)

        stop = self.callbacks.on_fit_start()
        for _ in range(epochs):
            if stop: break
            stop = self.callbacks.on_epoch_start()
            if stop: break
            for input, target in dataloader:
                stop = self.callbacks.on_batch_start()
                if stop: break
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(input, target, self.metrics)
                self.batch_loss = self.batch_metrics['loss']
                self.batch_loss.backward()
                stop = self.callbacks.before_step()
                if stop: break
                self.optimizer.step()
                stop = self.callbacks.on_batch_end()
                if stop: break
            else:
                stop = self.callbacks.on_epoch_end()
        self.callbacks.on_fit_end()
        return self.log

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):
        """Fit  model with inputs and targets.
        
        Arguments:
            input {np.array, tensor or tuple} -- Input (x) passed to net.
            target {np.array, tensor or tuple} -- Target (y) passed to loss function.
        
        Keyword Arguments:
            batch_size {int} -- Elemets in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
            **kwargs -- Passed to the 'make_dataloader' method. Set e.g. `torch_ds_dl to use
                the TensorDataset and DataLoader provided by torch instead of the torchtuples
                implementations.
    
        Returns:
            TrainingLogger -- Training log
        """
        dataloader = self.make_dataloader((input, target), batch_size, shuffle, num_workers, **kwargs)
        val_dataloader = val_data
        if type(val_data) in (list, tuple, TupleTree):
            val_dataloader = self.make_dataloader(val_data, val_batch_size, shuffle=False,
                                                  num_workers=num_workers, **kwargs)
        log = self.fit_dataloader(dataloader, epochs, callbacks, verbose, metrics, val_dataloader)
        return log

    @contextlib.contextmanager
    def _lr_finder(self, lr_min, lr_max, lr_range, n_steps, tolerance, verbose):
        lr_lower, lr_upper = lr_range
        path = make_name_hash('lr_finder_checkpoint')
        self.save_model_weights(path)
        try:
            self.optimizer.drop_scheduler()
            lr_finder = cb.LRFinder(lr_lower, lr_upper, n_steps, tolerance)
            yield lr_finder
        except Exception as e:
            self.load_model_weights(path)
            os.remove(path)
            self.optimizer = self.optimizer.reinitialize()
            self._init_train_log()
            raise e
        self.load_model_weights(path)
        os.remove(path)
        lr = lr_finder.get_best_lr(lr_min, lr_max)
        self.optimizer = self.optimizer.reinitialize(lr=lr)
        self._init_train_log()

    def lr_finder(self, input, target, batch_size=256, lr_min=1e-4, lr_max=1., lr_range=(1e-7, 10.),
                  n_steps=100, tolerance=np.inf, callbacks=None, verbose=False, num_workers=0,
                  shuffle=True, **kwargs):
        with self._lr_finder(lr_min, lr_max, lr_range, n_steps, tolerance, verbose) as lr_finder:
            if callbacks is None:
                callbacks = []
            callbacks.append(lr_finder)
            epochs = n_steps
            self.fit(input, target, batch_size, epochs, callbacks, verbose, num_workers,
                        shuffle, **kwargs)
        return lr_finder

    def lr_finder_dataloader(self, dataloader, lr_min=1e-4, lr_max=1., lr_range=(1e-7, 10.),
                             n_steps=100, tolerance=np.inf, callbacks=None, verbose=False):
        with self._lr_finder(lr_min, lr_max, lr_range, n_steps, tolerance, verbose) as lr_finder:
            if callbacks is None:
                callbacks = []
            callbacks.append(lr_finder)
            epochs = n_steps
            self.fit_dataloader(dataloader, epochs, callbacks, verbose)
        return lr_finder

    def score_in_batches(self, data, score_func=None, batch_size=8224, eval_=True, mean=True,
                         num_workers=0, shuffle=False, make_dataloader=None, numpy=True, **kwargs):
        """Used to score a dataset in batches.
        If score_func is None, this use the loss function.
        If make_dataloader is None, we use self.make_dataloader_predict, unless score_func is also
        None, in which we use self.make_dataloader.
        
        Arguments:
            data {np.array, tensor, tuple, dataloader} -- Data in the form a datloader, or arrarys/tensors.
        
        Keyword Arguments:
            score_func {func} -- Function used for scoreing. If None, we use self.loss. (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- Eval mode of the net. (default: {True})
            mean {bool} -- If True, we return the mean. (default: {True})
            num_workers {int} -- Number of workers for the dataloader. (default: {0})
            shuffle {bool} -- If the data should be shuffled (default: {False})
            make_dataloader {func} -- Function for making a dataloder.
                If None, we use make_dataloader_predict as long as score_func is not None. (default: {None})
            **kwargs -- Are passed to make_dataloader function.
        
        Returns:
            np.array -- Scores
        """
        if make_dataloader is None:
            if score_func is None:
                make_dataloader = self.make_dataloader
            else:
                make_dataloader = self.make_dataloader_predict
        if data.__class__ in (list, tuple, TupleTree):
            data = make_dataloader(data, batch_size, shuffle, num_workers, **kwargs)
        scores = self.score_in_batches_dataloader(data, score_func, eval_, mean, numpy)
        return scores
    
    def score_in_batches_dataloader(self, dataloader, score_func=None, eval_=True, mean=True, numpy=True):
        '''Score a dataset in batches.

        Parameters:
            dataloader: Dataloader:
            score_func: Function of (self, data) that returns a measure.
                If None, we get training loss.
            eval_: If net should be in eval mode.
            mean: If return mean or list with scores.
        '''
        if eval_:
            self.net.eval()
        batch_scores = []
        with torch.no_grad():
            for input, target in dataloader:
                if score_func is None:
                    # score = self.compute_loss(input, target)
                    score = self.compute_metrics(input, target, self.metrics)
                else:
                    warnings.warn(f"score_func {score_func} probably doesn't work... Not implemented")
                    score = score_func(self, input, target)
                batch_scores.append(score)
        if eval_:
            self.net.train()
        if type(batch_scores[0]) is dict:
            scores = defaultdict(list)
            for bs in batch_scores:
                for name, score in bs.items():
                    scores[name].append(score)
            scores = {name: torch.tensor(score) for name, score in scores.items()}
            if mean:
                scores = {name: score.mean() for name, score in scores.items()}
            if numpy:
                # scores = {name: tuplefy(score).to_numpy()[0] for name, score in scores.items()}
                scores = {name: score.item() for name, score in scores.items()}
            return scores
        if mean:
            batch_scores = [score.item() for score in batch_scores]
            return np.mean(batch_scores)
        return batch_scores

    def predict(self, input, batch_size=8224, numpy=None, eval_=True,
                grads=False, to_cpu=False, num_workers=0):
        """Get predictions from 'input'.
        
        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        if numpy is None:
            numpy = tuplefy(input).type() is not torch.Tensor
        dataloader = self.make_dataloader_predict(input, batch_size, shuffle=False,
                                                  num_workers=num_workers)
        preds = self.predict_dataloader(dataloader, numpy, eval_, grads, to_cpu)
        return preds

    def predict_dataloader(self, dataloader, numpy=True, eval_=True, grads=False, to_cpu=False):
        """Get predictions from dataloader.
        
        Arguments:
            dataloader {DataLoader} -- Dataloader with inputs to net.
        
        Keyword Arguments:
            numpy {bool} -- If 'False', tensor is returned (default: {True})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
        
        Returns:
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        if hasattr(self, 'fit_info') and (self.make_dataloader is self.make_dataloader_predict):
            # input = tuplefy(dataloader.dataset[0])
            data = _get_element_in_dataloader(dataloader)
            if data is not None:
                input = tuplefy(data)
                input_train = self.fit_info['input']
                if input.to_levels() != input_train['levels']:
                    raise RuntimeError("""The input from the dataloader is different from
                    the 'input' during trainig. Make sure to remove 'target' from dataloader.
                    Can be done with 'torchtuples.data.dataloader_input_only'.""")
                if input.shapes().apply(lambda x: x[1:]) != input_train['shapes']:
                    raise RuntimeError("""The input from the dataloader is different from
                    the 'input' during trainig. The shapes are different.""")

        if not eval_:
            warnings.warn("We still don't shuffle the data here... event though 'eval_' is True.")
        if eval_:
            self.net.eval()
        with torch.set_grad_enabled(grads):
            preds = []
            for input in dataloader:
                input = tuplefy(input).to_device(self.device)
                preds_batch = tuplefy(self.net(*input))
                if numpy or to_cpu:
                    preds_batch = preds_batch.to_device('cpu')
                preds.append(preds_batch)
        if eval_:
            self.net.eval()
        preds = tuplefy(preds).cat()
        if numpy:
            preds = preds.to_numpy()
        if len(preds) == 1:
            preds = preds[0]
        return preds

    def save_model_weights(self, path, **kwargs):
        '''Save the model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.save method.
        '''
        return torch.save(self.net.state_dict(), path, **kwargs)

    def load_model_weights(self, path, **kwargs):
        '''Load model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.load method.
        '''
        self.net.load_state_dict(torch.load(path, **kwargs))

    def save_net(self, path, **kwargs):
        """Save self.net to file (e.g. net.pt).
        
        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.save
        
        Returns:
            None
        """
        return torch.save(self.net, path, **kwargs)

    def load_net(self, path, **kwargs):
        """Load net from file (e.g. net.pt), and set as self.net
        
        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.load
        
        Returns:
            torch.nn.Module -- self.net
        """
        self.net = torch.load(path, **kwargs)
        if hasattr(self, 'device'):
            self.net.to(self.device)
        return self.net

def _get_element_in_dataloader(dataloader):
    dataset = dataloader.dataset
    try:
        return dataset[:2]
    except:
        pass
    try:
        return dataset[[0, 1]]
    except:
        pass
    try:
        return dataloader.collate_fn([dataset[0], dataset[1]])
    except:
        pass
    return None
