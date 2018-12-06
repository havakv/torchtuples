from torch import optim
import pyth.callbacks as cb

class OptimWrap(cb.CallbackHandler):
    def __init__(self, optimizer, callbacks=None):
        self.optimizer = optimizer
        if callbacks is None:
            callbacks = {}
        super().__init__(callbacks)
    
    def step(self, closure=None):
        return self.optimizer.step(closure)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def set(self, key, val):
        for param_group in self.optimizer.param_groups:
            param_group[key] = val
        # return self

    def set_wd(self, wd):
        """Set weight decay (wd not as in torch.optim)"""
        # self.wd = wd
        return self.set('wd', wd)

    def set_lr(self, lr):
        """Sets 'initial_lr' and 'lr' to value 'lr'"""
        self.set('initial_lr', lr)
        self.set('lr', lr)
        # return self

    def set_momentum(self, momentum):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (momentum, param_group['betas'][1])
        elif 'momentum' in first_gr:
            self.set('momentum', momentum)
        else:
            raise ValueError("No momentum found")
        # return self

    def set_beta(self, beta):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (param_group['betas'][0], beta)
        elif 'alpha' in first_gr:
            self.set('alpha', beta)
        else:
            raise ValueError("No beta found")
        # return self
    
    def drop_scheduler(self):
        pass

    def reinitialize(self, params=None, **kwargs):
        print('Optimizer is not reinitialized. Need to do this manually')
        return NotImplemented


class OptimizerW(OptimWrap):
    optim_func = NotImplemented
    init_args = NotImplemented

    def __init__(self, callbacks=None, wd=0, wd_normalize=False,
                 nb_epochs=None, params=None):
        callbacks = callbacks if callbacks else {}
        if 'weight_decay' in callbacks.keys():
            raise ValueError("weight_decay allreday exists")
        self.weight_decay = cb.WeightDecay(wd, wd_normalize, nb_epochs)
        callbacks['weight_decay'] = self.weight_decay
        super().__init__(None, callbacks)
        # self.optimizer = None
        if params is not None:
            self.init_optimizer(params)

    def __call__(self, params):
        self.init_optimizer(params)
        return self

    def init_optimizer(self, params):
        call_args = set(self.optim_func.__init__.__code__.co_varnames[1:])
        init_args = {name: self.init_args[name] for name in (call_args & self.init_args.keys())}
        self.optimizer = self.optim_func(params, **init_args)
        self.set_wd(self.init_args['wd'])

    def reinitialize(self, params=None, **kwargs):
        if params is None:
            if hasattr(self, 'model'):
                params = self.model.net.parameters()
        init_args = self.init_args.copy()
        init_args.update(kwargs)
        return self._constructor(**init_args, params=params)

    @property
    def _constructor(self):
        raise NotImplementedError


class AdamW(OptimizerW):
    optim_func = optim.Adam
    def __init__(self, lr=1e-3, betas=(0.9, 0.99), wd=0, wd_normalize=False,
                 nb_epochs=None, eps=1e-8, params=None):
        self.init_args = dict(lr=lr, betas=betas, eps=eps, wd=wd, wd_normalize=wd_normalize,
                              nb_epochs=nb_epochs)
        super().__init__(None, wd, wd_normalize, nb_epochs, params)

    @property
    def _constructor(self):
        return AdamW


class AdamWR(OptimizerW):
    optim_func = optim.Adam

    def __init__(self, lr=1e-3, betas=(0.9, 0.99), wd=0, wd_normalize=True, 
                 cycle_len="epoch", cycle_multiplier=2, eta_min=0,
                 params=None, eps=1e-8):
        self.init_args = dict(lr=lr, betas=betas, eps=eps, wd=wd, wd_normalize=wd_normalize,
                              cycle_len=cycle_len, cycle_multiplier=cycle_multiplier,
                              eta_min=eta_min)
        self.lr_scheduler = cb.LRCosineAnnealing(cycle_len, cycle_multiplier, eta_min)
        callbacks = {'lr_scheduler': self.lr_scheduler}
        nb_epochs = self.lr_scheduler.get_cycle_len if wd_normalize else None
        super().__init__(callbacks, wd, wd_normalize, nb_epochs, params)
    
    def drop_scheduler(self):
        self.callbacks.pop('lr_scheduler')
        self.weight_decay.nb_epochs = 1

    @property
    def _constructor(self):
        return AdamWR

