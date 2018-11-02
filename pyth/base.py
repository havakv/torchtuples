
'''Base models.
'''
from collections import OrderedDict
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset
from .data import DataLoaderSlice, DatasetTuple
from . import callbacks as cb

class Model(object):
    '''Abstract base model.

    Parameters:
        net: Pytorch Module.
        optimizer: Torch optimizer. If None Adam with default.
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `int`: used that gpu: torch.device('cuda:<device>').
            If `string`: string is passed to torch.device(`string`).
    '''
    def __init__(self, net, loss, optimizer=None, device=None, net_predict=None):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer if optimizer else optim.Adam(self.net.parameters())

        self.device = self._device_from__init__(device)
        self.net.to(self.device)
        self.net_predict = net_predict if net_predict else self.net
        self.net_predict.to(self.device)

        self.train_loss = cb.MonitorTrainLoss()
        self.log = cb.TrainingLogger()
        self.log.monitors = OrderedDict(train_loss=self.train_loss)
    
    @staticmethod
    def _device_from__init__(device):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device.__class__ is str:
            device = torch.device(device)
        elif device.__class__ is int:
            device = torch.device('cuda:{}'.format(device))
        else:
            if device.__class__ is not torch.device:
                raise ValueError('Argument `device` needs to be None, string, or torch.device object.')
        return device
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def _setup_train_info(self, dataloader, verbose, callbacks):
        self.fit_info = {'batches_per_epoch': len(dataloader)}

        self.log.verbose = verbose
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.CallbacksList([self.train_loss] + callbacks + [self.log])
        self.callbacks.give_model(self)

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True):
        self._setup_train_info(dataloader, verbose, callbacks)
        self.callbacks.on_fit_start()
        for _ in range(epochs):
            for input, target in dataloader:
                self.optimizer.zero_grad()
                self.batch_loss = self.compute_loss(input, target)
                self.batch_loss.backward()
                stop_signal = self.callbacks.before_step()
                if stop_signal:
                    raise RuntimeError('Stop signal in before_step().')
                self.optimizer.step()
                self.callbacks.on_batch_end()
            stop_signal = self.callbacks.on_epoch_end()
            if stop_signal:
                break
        return self.log
    
    def fit_tensor(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
                    num_workers=0):
        """Fit  model with inputs and targets.
        
        Arguments:
            input {tensor or tuple} -- Input (x) passed to net.
            target {tensor or tuple} -- Target (y) passed to loss function.
        
        Keyword Arguments:
            batch_size {int} -- Elemets in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
        """
        # if input.__class__ is torch.Tensor:
        #     input = (input,)
        # if target.__class__ is torch.Tensor:
        #     target = (target,)
        # dataset = DatasetTuple(input, target)
        # dataloader = DataLoaderSlice(dataset, batch_size, shuffle=True, num_workers=num_workers)
        dataloader = tensor_to_dataloader((input, target), batch_size, shuffle=True,
                                          num_workers=num_workers)
        return self.fit_dataloader(dataloader, epochs, callbacks, verbose)

    def fit_numpy(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
                  num_workers=0):
        """Fit model with inputs and targets.
        
        Arguments:
            input {array or tuple} -- Input (x) passed to net.
            target {array or tuple} -- Target (y) passed to loss function.
        
        Keyword Arguments:
            batch_size {int} -- Elemets in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
        """
        # input = numpy_to_tensor(input)
        # target = numpy_to_tensor(target)
        input, target = numpy_to_tensor((input, target))
        return self.fit_tensor(input, target, batch_size, epochs, callbacks, verbose, num_workers)
    
    def score_in_batches(self, data, score_func=None, batch_size=1028, eval_=True, mean=True,
                         num_workers=0, shuffle=False):
        if data.__class__ not in (list, tuple):
            return self.score_in_batches_dataloader(data, score_func, eval_, mean)
        input, target = data
        object_class = class_of(data)
        if object_class is torch.Tensor:
            return self.score_in_batches_tensor(input, target, score_func, batch_size,
                                                eval_, mean, num_workers, shuffle)
        elif object_class is np.ndarray:
            return self.score_in_batches_numpy(input, target, score_func, batch_size,
                                                eval_, mean, num_workers, shuffle)
        raise ValueError("Need `data` to be a dataloader or contain np.arrays or torch tensors.")
    
    def score_in_batches_numpy(self, input, target, score_func=None, batch_size=1028,
                                eval_=True, mean=True, num_workers=0, shuffle=False):
        input, target = numpy_to_tensor((input, target))
        return self.score_in_batches_tensor(input, target,score_func, batch_size,
                                            eval_, mean, num_workers, shuffle)

    def score_in_batches_tensor(self, input, target, score_func=None, batch_size=1028,
                                eval_=True, mean=True, num_workers=0, shuffle=False):
        dataloader = tensor_to_dataloader((input, target), batch_size, shuffle=shuffle,
                                          num_workers=num_workers)
        return self.score_in_batches_dataloader(dataloader, score_func, eval_, mean)
    
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
                    score = score_func(self, input, target)
                batch_scores.append(score)
        if eval_:
            self.net.train()
        if mean:
            batch_scores = [score.item() for score in batch_scores]
            return np.mean(batch_scores)
        return batch_scores

    def compute_loss(self, input, target):
        # input, target = data
        input = self._to_device(input)
        target = self._to_device(target)
        out = self.net(*input)
        out = tuple_if_tensor(out)
        return self.loss(*out, *target)
    
    def _to_device(self, data):
        data = to_device(data, self.device)
        return tuple_if_tensor(data)
    
    def predict_func_dataloader(self, dataloader, func=None, return_numpy=True, eval_=True, grads=False, move_to_cpu=False):
        '''Get func(X) for dataloader.

        Parameters:
            dataloader: Pytorch dataloader.
            func: Pytorch module.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set `fun` in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves `fun` modes as they are.
            grads: If gradients should be computed.
            move_to_cpu: For large data set we want to keep as torch.Tensors we need to
                move them to the cpu.
        '''
        #########################3
        # Need to fix this so it understands which part of dataloader is x and y
        ######################
        if func is None:
            func = self.net_predict
        if eval_:
            func.eval()
        with torch.set_grad_enabled(grads):
            preds = [self._predict_move_between_devices(func, x, return_numpy, move_to_cpu) 
                     for x in dataloader]
        if eval_:
            func.train()
        
        if preds[0].__class__ is torch.Tensor:
            preds = torch.cat(preds)
        else:
            preds = [torch.cat(sub) for sub in (zip(*preds))]

        if return_numpy:
            if preds.__class__ is torch.Tensor:
                return preds.numpy()
            else:
                return [sub.numpy() for sub in preds]
        return preds
    
    def _predict_move_between_devices(self, func, x, return_numpy, move_to_cpu):
        preds = func(*self._to_device(x))
        if return_numpy or move_to_cpu:
            if preds.__class__ is torch.Tensor:
                preds = preds.cpu()
            else:
                return [sub.cpu() for sub in preds]
        return preds

    def predict_func_tensor(self, x, func=None, batch_size=8224, return_numpy=False, eval_=True,
                            grads=False, move_to_cpu=False):
        '''Get func(X) for a tensor (or list of tensors) x.

        Parameters:
            x: Tensor or list of tensors with covariates.
            func: Pytorch module.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set `fun` in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves `fun` modes as they are.
            move_to_cpu: For large data set we want to keep as torch.Tensors we need to
                move them to the cpu.
        '''
        dataset = TensorDataset(*[x])
        dataloader = DataLoaderSlice(dataset, batch_size)
        return self.predict_func_dataloader(dataloader, func, return_numpy, eval_, grads,
                                            move_to_cpu)

    def predict_func_numpy(self, x, func=None, batch_size=8224, return_numpy=True, eval_=True,
                           grads=False, move_to_cpu=False):
        '''Get func(X) for a numpy array x.

        Parameters:
            X: Numpy matrix with with covariates.
            func: Pytorch module.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set `fun` in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves `fun` modes as they are.
            move_to_cpu: For large data set we want to keep as torch.Tensors we need to
                move them to the cpu.
        '''
        x = numpy_to_tensor(x)
        return self.predict_func_tensor(x, func, batch_size, return_numpy, eval_, grads,
                                        move_to_cpu)

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


def class_of(data):
    if data.__class__ not in (list, tuple):
        return data.__class__
    classes = [class_of(sub) for sub in data]
    if classes.count(classes[0]) != len(classes):
        raise ValueError("All objects in 'data' doest have the same class.")
    return classes[0]

def numpy_to_tensor(data):
    if data.__class__ in (list, tuple):
        return tuple(numpy_to_tensor(sub) for sub in data)
    return torch.from_numpy(data)#.float()
    
def tensor_to_dataloader(data, batch_size, shuffle, num_workers):
    if class_of(data) is not torch.Tensor:
        raise ValueError(f"Need 'data' to be tensors, not {class_of(data)}.")
    if data.__class__ is torch.Tensor:
        data = (data,)
    dataset = DatasetTuple(data)
    dataloader = DataLoaderSlice(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def numpy_to_dataloader(data, batch_size, shuffle, num_workers):
    data = numpy_to_tensor(data)
    return tensor_to_dataloader(data, batch_size, shuffle=shuffle, num_workers=num_workers)

def to_device(data, device):
    if class_of(data) is not torch.Tensor:
        raise ValueError(f"Need 'data' to be tensors, not {class_of(data)}.")
    if data.__class__ is torch.Tensor:
        return data.to(device)
    return tuple(to_device(sub, device) for sub in data)

def tuple_if_tensor(data):
    if data.__class__ is torch.Tensor:
        data = (data,)
    return data
