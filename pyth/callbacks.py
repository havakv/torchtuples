'''
Callbacks.
'''
import warnings
import time
from collections import OrderedDict, defaultdict
import math
import numpy as np
try:
    import pandas as pd
except:
    pass
import torch
import pyth
from . import lr_scheduler 

class CallbackHandler:
    def __init__(self, callbacks):
        self.callbacks = OrderedDict()
        if type(callbacks) in (list, tuple):
            cb_as_dict = OrderedDict()
            for cb in callbacks:
                cb_as_dict[self._make_name(cb)] = cb
            callbacks = cb_as_dict
        self.callbacks = OrderedDict(callbacks)
        self.model = None

    def _make_name(self, obj):
        name = obj.__class__.__name__
        i = 0
        while name in self.callbacks.keys():
            name = name + '_' + str(i)
            i += 1
        return name
    
    def __getitem__(self, name):
        return self.callbacks[name]
    
    def __setitem__(self, name, callback):
        self.append(callback, name)

    def append(self, callback, name=None):
        if self.model is None:
            raise RuntimeError("Can only call append after the callback has received the model.")
        callback.give_model(self.model)
        if name is None:
            name = self._make_name(callback)
        assert name not in self.callbacks.keys(), f"Duplicate name: {name}"
        self.callbacks[name] = callback
    
    def items(self):
        return self.callbacks.items()
    
    def keys(self):
        return self.callbacks.keys()
    
    def values(self):
        return self.callbacks.values()
    
    def __len__(self):
        return len(self.callbacks)

    def apply_callbacks(self, func):
        stop_signal = False
        for c in self.values():
            stop = func(c)
            stop = stop if stop else False
            stop_signal += stop
        return stop_signal

    def give_model(self, model):
        self.model = model
        stop_signal = self.apply_callbacks(lambda x: x.give_model(model))
        return stop_signal

    def on_fit_start(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_fit_start())
        return stop_signal

    def before_step(self):
        stop_signal = self.apply_callbacks(lambda x: x.before_step())
        return stop_signal

    def on_batch_end(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_batch_end())
        return stop_signal

    def on_epoch_end(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_epoch_end())
        return stop_signal


class TrainingCallbackHandler(CallbackHandler):
    '''Object for holding all callbacks.
    '''
    def __init__(self, optimizer, train_metrics, log, val_metrics=None, callbacks=None):
        self.callbacks = OrderedDict()
        self.callbacks['optimizer'] = optimizer
        self.callbacks['train_metrics'] = train_metrics
        if val_metrics:
            self.callbacks['val_metrics'] = val_metrics
        self.callbacks['log'] = log

        if type(callbacks) in (list, tuple):
            cb_as_dict = OrderedDict()
            for cb in callbacks:
                cb_as_dict[self._make_name(cb)] = cb
            callbacks = cb_as_dict
        if callbacks is not None:
            callbacks = OrderedDict(callbacks)
            for name, cb in callbacks.items():
                assert name not in self.callbacks.keys(), f"Duplicate name: {name}"
                self.callbacks[name] = cb

        self.callbacks.move_to_end('log')
        self.model = None

    def append(self, callback, name=None):
        super().append(callback, name)
        self.callbacks.move_to_end('log')


class Callback(object):
    '''Abstract class for callbacks.
    '''
    def give_model(self, model):
        self.model = model

    def on_fit_start(self):
        pass

    def before_step(self):
        """Called after loss.backward(), but before optim.step()."""
        stop_signal = False
        return stop_signal

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        stop_signal = False
        return stop_signal


# class PlotProgress(Callback):
#     '''Plott progress

#     Parameters:
#         monitor: Dict with names and Moniro objects.
#         filename: Filename (without ending).
#         type: If 'altair' plot with altair.
#             If not, type is given as fileending to matplotlib.
#     '''
#     def __init__(self, monitor, filename='progress', type='svg',
#                  style='fivethirtyeight'):
#         super().__init__()
#         self.filename = filename
#         self._first = True
#         assert monitor.__class__ in [dict, OrderedDict], 'for now we need dict'
#         self.monitor = monitor
#         self.type = type
#         self.style = style

#     def on_epoch_end(self):
#         if self._first:
#             self._first = False
#             return False

#         with plt.style.context(self.style):
#             self.to_pandas().plot()
#             plt.savefig(self.filename+'.'+self.type)
#         plt.close('all')
#         return False


#     def to_pandas(self, naming='prefix'):
#         '''Get data in dataframe.

#         Parameters:
#             naming: Put name of metrix as prefix of suffix.
#         '''
#         warnings.warn('Need to updata this one')
#         df = pd.DataFrame()
#         if self.monitor.__class__ in [dict, OrderedDict]:
#             for name, mm in self.monitor.items():
#                 d = mm.to_pandas()
#                 if naming == 'suffix':
#                     df = df.join(d, how='outer', rsuffix=name)
#                     continue
#                 if naming == 'prefix':
#                     d.columns = [name+'_'+c for c in d.columns]
#                 df = df.join(d, how='outer')
#         return df


class TrainingLogger(Callback):
    def __init__(self, verbose=1):
        self.epoch = 0
        self.epochs = []
        self.loss = []
        self._verbose = verbose
    
    @property
    def monitors(self):
        return self._monitors

    @monitors.setter
    def monitors(self, monitor_dict):
        self._monitors = monitor_dict

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def on_fit_start(self):
        self.prev_time = time.time()

    def on_epoch_end(self):
        self.epochs.append(self.epoch)
        if self.verbose:
            self.print_on_epoch_end()
        self.epoch += 1
        return False
    
    def print_on_epoch_end(self):
        new_time = time.time()
        string = 'Epoch: %d,\ttime: %d sec,' % (self.epoch, new_time - self.prev_time)
        print(string + self.get_measures())
        self.prev_time = new_time

    def get_measures(self):
        measures = self.monitors
        if self.verbose.__class__ in [dict, OrderedDict]:
            measures = OrderedDict(measures, **self.verbose)
        string = ''
        for prefix, mm in measures.items():
            for name, score in mm.to_pandas().iloc[-1].items(): # slow but might not matter
                if (score is not None) and (np.isnan(score) == False):
                    string += '\t%s:' % (prefix + name)
                    string += ' %.4f,' % score
        return string[:-1]

    def to_pandas(self, colnames=None):
        '''Get data in dataframe.
        '''
        assert colnames is None, 'Not implemented'
        dfs = []
        for prefix, mm in self.monitors.items():
            df = mm.to_pandas()
            df.columns = prefix + df.columns
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df

    def plot(self, colnames=None, **kwargs):
        return self.to_pandas(colnames).plot(**kwargs)


class EarlyStopping(Callback):
    '''Stop training when monitored quantity has stopped improving.
    Takes a Monitor object and runs it as a callback.
    Use first metric in mm_obj to determine early stopping.

    Parameters:
        mm_obj: Monitor object, where first metric is used for early stopping.
            E.g. MonitorSurvival(df_val, 'cindex').
        minimize: If we are to minimize or maximize monitor.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        model_file_path: If spesified, the model weights will be stored whever a better score
            is achieved.
    '''
    def __init__(self, mm_obj, minimize=True, min_delta=0, patience=10, model_file_path=None):
        if True:
            raise NotImplementedError
        self.mm_obj = mm_obj
        self.minimize = minimize
        self.min_delta = min_delta
        self.patience = patience
        self.model_file_path = model_file_path
        self.cur_best = np.inf if self.minimize else -np.inf
        self.scores = []
        self.n = 0

    def on_epoch_end(self):
        score = self.mm_obj.scores[0][-1]
        self.scores.append(score)

        if self.minimize:
            if score < (self.cur_best - self.min_delta):
                self.cur_best = score
                self.n = -1
        else:
            if score > (self.cur_best + self.min_delta):
                self.cur_best = score
                self.n = -1
        self.n += 1

        if (self.n == 0) and (self.model_file_path is not None):
            self.model.save_model_weights(self.model_file_path)

        stop_signal = True if self.n >= self.patience else False
        return stop_signal


class MonitorMetrics(Callback):
    def __init__(self):
        self.scores = dict()
        self.epoch = -1

    def on_epoch_end(self):
        self.epoch += 1
    
    def append_score(self, name, val):
        scores = self.scores.get(name, {'epoch': [], 'score': []})
        scores['epoch'].append(self.epoch)
        scores['score'].append(val)
        self.scores[name] = scores

    def to_pandas(self):
        '''Return scores as a pandas dataframe'''
        scores = [pd.Series(score['score'], index=score['epoch']).rename(name)
                  for name, score in self.scores.items()]
        scores = pd.concat(scores, axis=1)
        if type(scores) is pd.Series:
            scores = scores.to_frame()
        return scores


class _MonitorFitMetricsTrainData(MonitorMetrics):
    def __init__(self, per_epoch=1):
        super().__init__()
        self.per_epoch = per_epoch

    def on_fit_start(self):
        self.batch_metrics = defaultdict(list)
    
    def on_batch_end(self):
        for name, score in self.model.batch_metrics.items():
            self.batch_metrics[name].append(score.item())

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch != 0:
            return False
        for name, vals in self.batch_metrics.items():
            self.append_score(name, np.mean(vals))
        return False


class MonitorFitMetrics(MonitorMetrics):
    def __init__(self, dataloader=None, per_epoch=1):
        super().__init__()
        self.dataloader = dataloader
        self.per_epoch = per_epoch

    @property
    def dataloader(self):
        return self._dataloader
    
    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch != 0:
            return False
        if self.dataloader is None:
            scores = {name: np.nan for name in self.model.metrics.keys()}
        else:
            scores = self.model.score_in_batches_dataloader(self.dataloader)
        for name, val in scores.items():
            self.append_score(name, val)
        return False


class MonitorTrainMetrics(Callback):
    '''Monitor metrics for training loss.

    Parameters:
        per_epoch: How often to calculate.
    '''
    @property
    def scores(self):
        return self.model.train_metrics.scores
    
    def to_pandas(self):
        '''Return scores as a pandas dataframe'''
        scores = [pd.Series(score[name], index=score['epoch']).rename(name)
                  for name, score in self.scores.items()]
        scores = pd.concat(scores, axis=1)
        if type(scores) is pd.Series:
            scores = scores.to_frame()
        return scores


# class MonitorXy(MonitorBase):
#     '''Monitor metrics for classification and regression.
#     Same as MonitorBase but we input a pair, X, y instead of data.

#     For survival methods, see e.g. MonitorCox.

#     Parameters:
#         X: Numpy array with features.
#         y: Numpy array with labels.
#         monitor_funcs: Function, list, or dict of functions giving quiatities that should
#             be monitored.
#             The function takes argumess (df, preds) and should return a score.
#         batch_size: Batch size used for calculating the scores.
#         **kwargs: Can be passed to predict method.
#     '''
#     def __init__(self, X, y, monitor_funcs, per_epoch=1, batch_size=512, **kwargs):
#         self.X, self.y = X, y
#         self.kwargs = kwargs
#         super().__init__(monitor_funcs, per_epoch, batch_size)

#     def get_score_inputs(self):
#         '''This function should create arguments to the monitor function.
#         Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
#         '''
#         preds = self.model.predict(self.X, self.batch_size, **self.kwargs)
#         return self.y, preds


# class MonitorSklearn(MonitorXy):
#     '''Class for monitoring metrics of from sklearn metrics

#     Parameters:
#         X: Numpy array with features.
#         y: Numpy array with labels.
#         monitor: Name for method in sklearn.metrics.
#             E.g. 'log_loss', or pass sklearn.metrics.log_loss.
#             For additional parameter, specify the function with a lambda statemetn.
#                 e.g. {'accuracy': lambda y, p: metrics.accuracy_score(y, p > 0.5)}
#         batch_size: Batch size used for calculating the scores.
#         **kwargs: Can be passed to predict method.
#     '''
#     def __init__(self, X, y, monitor, per_epoch=1, batch_size=512, **kwargs):
#         if monitor.__class__ is str:
#             monitor = {monitor: monitor}
#         elif monitor.__class__ is list:
#             monitor = {mon if mon.__class__ is str else str(i): mon
#                        for i, mon in enumerate(monitor)}

#         monitor = {name: getattr(metrics, mon) if mon.__class__ == str else mon
#                    for name, mon in monitor.items()}
#         super().__init__(X, y, monitor, per_epoch, batch_size)


class ClipGradNorm(Callback):
    '''Callback for clipping gradients.
    
    See torch.nn.utils.clip_grad_norm_.

    Parameters:
        net: Network wtih parameters() function.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    '''
    def __init__(self, net, max_norm, norm_type=2):
        self.net = net
        self.max_norm = max_norm
        self.norm_type = norm_type

    def before_step(self):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm, self.norm_type)
        stop_signal = False
        return stop_signal


class LRScheduler(Callback):
    '''Wrapper for pytorch.optim.lr_scheduler objects.

    Parameters:
        scheduler: A pytorch.optim.lr_scheduler object.
        mm_obj: Monitor object, where first metric is used for early stopping.
            E.g. MonitorSurvival(df_val, 'cindex').
    '''
    def __init__(self, scheduler, mm_obj):
        self.scheduler = scheduler
        self.mm_obj = mm_obj

    def on_epoch_end(self):
        score = self.mm_obj.scores[0][-1]
        self.scheduler.step(score)
        stop_signal = False
        return stop_signal


class LRSchedulerBatch(Callback):
    '''Wrapper for schedulers

    Parameters:
        scheduler: A scheduler, e.g. BatchCosineAnnealingLR()
    '''
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_fit_start(self):
        self.scheduler.step()  
    
    def on_batch_end(self):
        self.scheduler.step()
        return False


class LRCosineAnnealing(LRSchedulerBatch):
    def __init__(self, cycle_len="epoch", cycle_multiplier=2, eta_min=0):
        self.first_cycle_len = cycle_len
        self.cycle_multiplier = cycle_multiplier
        self.eta_min = eta_min
        scheduler = None
        super().__init__(scheduler)
    
    def on_fit_start(self):
        if self.first_cycle_len == "epoch":
            self.first_cycle_len = self.model.fit_info['batches_per_epoch']
        if not self.scheduler:
            scheduler = lr_scheduler.LRBatchCosineAnnealing(self.model.optimizer, self.first_cycle_len,
                                                            self.cycle_multiplier, self.eta_min,
                                                            keep_etas=True)
            self.scheduler = scheduler
        elif self.model.optimizer is not self.scheduler.optimizer:
            raise RuntimeError(
                "Changed optimizer, and we have not implemented cosine annealing for this")
        super().on_fit_start()
    
    def get_cycle_len(self):
        return self.scheduler.cycle_len
    
    def get_epochs_per_cycle(self):
        return self.get_cycle_len() / self.model.fit_info['batches_per_epoch']

    def get_etas(self):
        return self.scheduler.etas


class LRFinder(Callback):
    def __init__(self, lr_min=1e-7, lr_max=10., n_steps=100, tolerance=10.):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.n_steps = n_steps
        self.lowest_loss = np.inf
        self.tolerance = tolerance

    def on_fit_start(self):
        self.batch_loss = []
        self.scheduler = lr_scheduler.LRFinderScheduler(self.model.optimizer, self.lr_min,
                                                        self.lr_max, self.n_steps)
        self.scheduler.step()
    
    def on_batch_end(self):
        self.scheduler.step()
        batch_loss = self.model.batch_loss.item()
        self.batch_loss.append(batch_loss)
        if (batch_loss / self.lowest_loss) > self.tolerance:
            return True
        self.lowest_loss = min(self.lowest_loss, batch_loss)
        if self.scheduler.batch_iter == self.n_steps:
            return True
        return False
    
    def to_pandas(self, smoothed=0):
        res = pd.DataFrame(dict(train_loss=self.batch_loss),
                           index=self.lrs[:len(self.batch_loss)])
        res.index.name = 'lr'
        if smoothed:
            res = res.apply(_smooth_curve, beta=smoothed)
        return res
    
    def plot(self, logx=True, smoothed=0.98, **kwargs):
        res = self.to_pandas(smoothed)
        ylabel = 'bach_loss'
        if smoothed:
            ylabel = ylabel + ' (smoothed)'
        ax = res.plot(logx=True, **kwargs)
        ax.set_xlabel('lr')
        ax.set_ylabel(ylabel)
        return ax

    @property
    def lrs(self):
        return self.scheduler.lrs
    
    def get_best_lr(self, lower=1e-4, upper=1., _multiplier=10.):
        """Get suggestion for bets learning rate.
        It is beter to investigate the plot, but this might work too.
        
        Keyword Arguments:
            lower {float} -- Lower accepable learning rate (default: {1e-4})
            upper {float} -- Upper acceptable learning rate (default: {1.})
            _multiplier {float} -- See sorce code (default according to fast.ai) (default: {10})
        
        Returns:
            float -- Suggested best learning rate.
        """

        idx_lower = np.searchsorted(self.lrs, lower * _multiplier)
        idx_upper = np.searchsorted(self.lrs, upper * _multiplier, 'right')
        smoothed = _smooth_curve(self.batch_loss)[idx_lower:idx_upper]
        idx_min = np.argmin(smoothed)
        best_lr = self.lrs[idx_lower:idx_upper][idx_min] / _multiplier
        return best_lr

def _smooth_curve(vals, beta=0.98):
    """From fastai"""
    avg_val = 0
    smoothed = []
    for (i,v) in enumerate(vals):
        avg_val = beta * avg_val + (1-beta) * v
        smoothed.append(avg_val/(1-beta**(i+1)))
    return smoothed


class WeightDecay(Callback):
    """Same weight decay for all groups in the optimizer."""
    def __init__(self, weight_decay, normalized=False, nb_epochs=None):
        self.weight_decay = weight_decay
        self.normalized = normalized
        # if self.normalized:
        #     if not ((type(nb_epochs) is int) or callable(nb_epochs)):
        #         raise ValueError(f"Need nb_epochs to be callable or int, not {nb_epochs}")
        self.nb_epochs = nb_epochs
    
    def on_fit_start(self):
        self._batches_per_epoch = self.model.fit_info['batches_per_epoch']

    @property
    def nb_epochs(self):
        if type(self._nb_epochs) is int:
            return self._nb_epochs
        elif callable(self._nb_epochs):
            return self._nb_epochs()
        raise RuntimeError('nb_epochs needs to be callable or int')

    @nb_epochs.setter
    def nb_epochs(self, nb_epochs):
        if self.normalized:
            if not ((type(nb_epochs) is int) or callable(nb_epochs)):
                raise ValueError(f"Need nb_epochs to be callable or int, not {type(nb_epochs)}")
        self._nb_epochs = nb_epochs
    
    def _normalized_weight_decay(self):
        # if type(self.nb_epochs) is int:
        #     nb_epochs = self.nb_epochs
        # elif callable(self.nb_epochs):
        #     nb_epochs = self.nb_epochs()
        # else:
        #     raise RuntimeError('nb_epochs needs to be callable or int')
        norm_const = math.sqrt(1 / (self._batches_per_epoch * self.nb_epochs))
        return self.weight_decay * norm_const
    
    def before_step(self):
        # Weight decay out of the loss. After the gradient computation but before the step.
        if self.normalized:
            weight_decay = self._normalized_weight_decay()
        else:
            weight_decay = self.weight_decay
        for group in self.model.optimizer.param_groups:
            lr = group['lr']
            # alpha = group.get('initial_lr', 1.)
            alpha = group.get('initial_lr', lr)
            eta = lr / alpha
            for p in group['params']:
                if p.grad is not None:
                    p.data = p.data.add(-weight_decay * eta, p.data)
                    # p.data.mul_(1 - weight_decay * eta)
        return False


class EarlyStoppingCycle(Callback):
    '''Stop training when monitored quantity has not improved the last cycle.
    Takes a Monitor object that is also a callback.
    Use first metric in mm_obj to determine early stopping.

    Parameters:
        mm_obj: Monitor object, where first metric is used for early stopping.
            E.g. MonitorSurvival(df_val, 'cindex').
        get_score: Function for obtaining current scores. If None, we use validation loss.
        minimize: If we are to minimize or maximize monitor.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of cycles patience.
        model_file_path: If spesified, the model weights will be stored whever a better score
            is achieved.
    '''
    def __init__(self, lr_scheduler='optimizer', get_score=None, minimize=True, min_delta=0,
                 patience=1, min_cycles=4, model_file_path=None):
        self.get_score = get_score
        self.minimize = minimize
        self.min_delta = min_delta
        self.patience = patience
        self.min_cycles = min_cycles
        self.model_file_path = model_file_path
        self.lr_scheduler = lr_scheduler
        self.cur_best = np.inf if self.minimize else -np.inf
        self.cur_best_cycle_nb = None

    def on_fit_start(self):
        if self.get_score is None:
            self.get_score = lambda: self.model.val_metrics.scores['loss']['score'][-1]
        if self.lr_scheduler == 'optimizer':
            self.lr_scheduler = self.model.optimizer.lr_scheduler
        
    def on_epoch_end(self):
        etas = self.lr_scheduler.get_etas()
        cycle_nb = etas.count(1.) - 1
        score = self.get_score()

        if self.minimize:
            if score < (self.cur_best - self.min_delta):
                self.cur_best = score
                self.cur_best_cycle_nb = cycle_nb
        else:
            if score > (self.cur_best + self.min_delta):
                self.cur_best = score
                self.cur_best_cycle_nb = cycle_nb

        if (score == self.cur_best) and (self.model_file_path is not None):
            self.model.save_model_weights(self.model_file_path)

        stop_signal = ((cycle_nb > (self.cur_best_cycle_nb + self.patience)) and 
                       (cycle_nb >= self.min_cycles))
        return stop_signal
