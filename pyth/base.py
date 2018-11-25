
"""Model for fitting torch models.
"""
from collections import OrderedDict
import warnings
import numpy as np  # shoul be remved
import torch
import pyth.callbacks as cb
from pyth.optim import AdamW
from pyth.tuple import tuplefy, Tuple, make_dataloader


class Model(object):
    """Train torch models using dataloaders, tensors or np.arrays.
    
    Arguments:
        net {torch.nn.Module} -- A torch module.
    
    Keyword Arguments:
        loss {function} -- Set function that is used for training 
            (e.g. binary_cross_entropy for torch) (default: {None})
        optimizer {Optimizer} -- A torch optimizer or similar.
            if 'None' set to pyth.optim.AdamW (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferrably pass a torch.device object.
            If 'None': use default gpu if avaiable, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    Example simple model:
    ---------------------
    from pyth import Model
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
        self.optimizer = optimizer if optimizer else AdamW(self.net.parameters())

        self.device = self._device_from__init__(device)
        self.net.to(self.device)
        # self.net_predict = net_predict if net_predict else self.net
        # self.net_predict.to(self.device)

        self.train_loss = cb.MonitorTrainLoss()
        self.log = cb.TrainingLogger()
        self.log.monitors = OrderedDict(train_loss=self.train_loss)
    
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
    
    def _setup_train_info(self, dataloader, verbose, callbacks):
        self.fit_info = {'batches_per_epoch': len(dataloader)}
        def _tuple_info(tuple_):
            tuple_ = tuplefy(tuple_)
            return {'levels': tuple_.to_levels(), 'shapes': tuple_.shapes().apply(lambda x: x[1:])}
        input, target = next(iter(dataloader))
        self.fit_info['input'] = _tuple_info(input)
        self.fit_info['target'] = _tuple_info(target)

        self.log.verbose = verbose
        self.callbacks = cb.CallbackHandler(self.train_loss, self.log, callbacks)
        self.callbacks.give_model(self)

    def compute_loss(self, input, target):
        """Function for computing loss.
        Is rather general, but can be reimpliemented by sub classes.
        
        Arguments:
            input {tensor or tuple} -- This should be passed to self.net.
            target {tensor or tuple} -- This is the targets that should be used in self.loss.
        
        Returns:
            tensor -- Results of self.loss()
        """
        if self.loss is None:
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        input = self._to_device(input)
        target = self._to_device(target)
        out = self.net(*input)
        out = tuplefy(out)
        return self.loss(*out, *target)
    
    def _to_device(self, data):
        """Move data to self.device.
        
        Arguments:
            data {tensor or tuple} -- Data
        
        Returns:
            tensor or tuple -- Data moved to device.
        """
        return tuplefy(data).to_device(self.device)

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True):
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
        self._setup_train_info(dataloader, verbose, callbacks)
        stop_signal = self.callbacks.on_fit_start()
        if stop_signal:
            raise RuntimeError('Got stop_signal from callback before fit starts')
        for _ in range(epochs):
            for input, target in dataloader:
                self.optimizer.zero_grad()
                self.batch_loss = self.compute_loss(input, target)
                self.batch_loss.backward()
                stop_signal += self.callbacks.before_step()
                if stop_signal:
                    raise RuntimeError('Stop signal in before_step().')
                self.optimizer.step()
                stop_signal += self.callbacks.on_batch_end()
                if stop_signal:
                    break
            else:
                stop_signal += self.callbacks.on_epoch_end()
            if stop_signal:
                break
        return self.log

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True):
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
    
        Returns:
            TrainingLogger -- Training log
        """
        dataloader = make_dataloader((input, target), batch_size, shuffle, num_workers)
        log = self.fit_dataloader(dataloader, epochs, callbacks, verbose)
        return log

    # legacy
    fit_numpy = fit
    fit_tensor = fit

    def score_in_batches(self, data, score_func=None, batch_size=8224, eval_=True, mean=True,
                         num_workers=0, shuffle=False):
        if data.__class__ in (list, tuple, Tuple):
            data = make_dataloader(data, batch_size, shuffle, num_workers)
        scores = self.score_in_batches_dataloader(data, score_func, eval_, mean)
        return scores
    
    def score_in_batches_dataloader(self, dataloader, score_func=None, eval_=True, mean=True):
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
                    score = self.compute_loss(input, target)
                else:
                    warnings.warn(f"score_func {score_func} probably doesn't work... Not implemented")
                    score = score_func(self, input, target)
                batch_scores.append(score)
        if eval_:
            self.net.train()
        if mean:
            batch_scores = [score.item() for score in batch_scores]
            return np.mean(batch_scores)
        return batch_scores

    def predict(self, input, batch_size=8224, return_numpy=True, eval_=True,
                grads=False, move_to_cpu=False, num_workers=0):
        """Get predictions from 'input'.
        
        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            return_numpy {bool} -- If 'False', tensor is returned (default: {True})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            move_to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            [Tuple, np.ndarray or tensor] -- Predictions
        """
        dataloader = make_dataloader(input, batch_size, shuffle=False, num_workers=num_workers)
        preds = self.predict_dataloader(dataloader, return_numpy, eval_=True, grads=False,
                                        move_to_cpu=False)
        return preds

    def predict_dataloader(self, dataloader, return_numpy=True, eval_=True,
                           grads=False, move_to_cpu=False):
        """Get predictions from dataloader.
        
        Arguments:
            dataloader {DataLoader} -- Dataloader with inputs to net.
        
        Keyword Arguments:
            return_numpy {bool} -- If 'False', tensor is returned (default: {True})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            move_to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
        
        Returns:
            [Tuple, np.ndarray or tensor] -- Predictions
        """
        if hasattr(self, 'fit_info'):
            input = tuplefy(next(iter(dataloader)))
            input_train = self.fit_info['input']
            if input.to_levels() != input_train['levels']:
                raise RuntimeError("""The input from the dataloader is different from
                the 'input' during trainig. Make sure to remove target from dataloader""")
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
                if return_numpy or move_to_cpu:
                    preds_batch = preds_batch.to_device('cpu')
                preds.append(preds_batch)
        if eval_:
            self.net.eval()
        preds = tuplefy(preds).cat()
        if return_numpy:
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
            self.net.to_device(self.device)
        return self.net
