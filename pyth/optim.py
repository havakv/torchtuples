from torch import optim
import pyth.callbacks as cb


class AdamW(optim.Adam):
    """Not a full implementation of AdamW, but just calls reglar Adam
    with other defaults:
        - beta2 = 0.99.
        - weight_decay = 0 (this is L2 regularization and not weight decay).
        - amsgra=False.
    
    Use callback WeightDecay to get weight decay.
    
    Arguments:
        params {iterable} -- Iterable of parameters to optimize, or dicts
            defining parameter groups.
    
    Keyword Arguments:
        lr {int} -- Learning rate (default: {1e-3})
        betas {tuple} -- Coefficients used for computing running averages
            of gradient and its square  (default: {(0.9, 0.99)})
        eps {[type]} -- Term added to the denominator to improve
            numerical stability (default: {1e-8})
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        super().__init__(params, lr, betas, eps, weight_decay=0, amsgrad=False)

        
class OptimWrap(cb.SubCallbackHandler):
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
    
    def set(self, key, val):
        for param_group in self.optimizer.param_groups:
            param_group[key] = val
        return self

    def set_wd(self, wd):
        """Set weight decay (wd not as in torch.optim)"""
        # self.wd = wd
        return self.set('wd', wd)

    def set_lr(self, lr):
        """Sets 'initial_lr' and 'lr' to value 'lr'"""
        self.set('initial_lr', lr)
        self.set('lr', lr)
        return self

    def set_momentum(self, momentum):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (momentum, param_group['betas'][1])
        elif 'momentum' in first_gr:
            self.set('momentum', momentum)
        else:
            raise ValueError("No momentum found")
        return self

    def set_beta(self, beta):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (param_group['betas'][0], beta)
        elif 'alpha' in first_gr:
            self.set('alpha', beta)
        else:
            raise ValueError("No beta found")
        return self
    
    def drop_scheduler(self):
        pass

    def reinitialize(self, parameters=None, **kwargs):
        print('Optimizer is not reinitialized. Need to do this manually')

# class Optimizer(OptimWrap):
#     optim_func = NotImplemented
#     def __init__(self, optimizer, callbacks=None, wd=0, wd_normalize=True, 
#                  nb_epochs=None, parameters=None):
#         pass

#     def __call__(self, parameters):
#         self.init_optimizer(parameters)
#         return self

#     def init_optimizer(self, parameters):
#         call_args = set(self.optim_func.__init__.__code__.co_varnames[1:])
#         init_args = {name: self.init_args[name] for name in (call_args & self.init_args.keys())}
#         self.optimizer = self.optim_func(parameters, **init_args)
#         self.set_wd(self.init_args['wd'])

# class AdamWR(Optimizer):
class AdamWR(OptimWrap):
    def __init__(self, lr=1e-3, betas=(0.9, 0.99), wd=0, wd_normalize=True, 
                 cycle_len="epoch", cycle_multiplier=2, eta_min=0,
                 parameters=None, eps=1e-8):
        self.init_args = dict(lr=lr, betas=betas, eps=eps, wd=wd, wd_normalize=wd_normalize,
                              cycle_len=cycle_len, cycle_multiplier=cycle_multiplier,
                              eta_min=eta_min)
        self.optimizer = None
        if parameters is not None:
            self.init_optimizer(parameters)

        self.lr_scheduler = cb.LRCosineAnnealing(cycle_len, cycle_multiplier, eta_min)
        self.weight_decay = cb.WeightDecay(wd, wd_normalize, self.lr_scheduler.get_cycle_len)
        callbacks = {'lr_scheduler': self.lr_scheduler, 'weight_decay': self.weight_decay}
        # cb.SubCallbackHandler.__init__(self, callbacks)
        super().__init__(self.optimizer, callbacks)
    
    def __call__(self, parameters):
        self.init_optimizer(parameters)
        return self

    def drop_scheduler(self):
        self.callbacks.pop('lr_scheduler')
        self.weight_decay.nb_epochs = 1

    def init_optimizer(self, parameters):
        optim_func = optim.Adam
        # init_args = self.init_args.copy()
        call_args = set(optim_func.__init__.__code__.co_varnames[1:])
        init_args = {name: self.init_args[name] for name in (call_args & self.init_args.keys())}
        self.optimizer = optim.Adam(parameters, **init_args)
        self.set_wd(self.init_args['wd'])

    @property
    def _constructror(self):
        return AdamWR

    def reinitialize(self, parameters=None, **kwargs):
        if parameters is None:
            if hasattr(self, 'model'):
                parameters = self.model.net.parameters()
        init_args = self.init_args.copy()
        init_args.update(kwargs)
        return self._constructror(**self.init_args, parameters=parameters)















# class AdamW(Optimizer):
#     """Implements AdamW algorithm.

#     Modified from pytorch.optimizer.Adam

#     lr corresponds to max lr
#     """

#     def __init__(self, params, lr, lr_min=1e-5, beta1=0.9, beta2=0.99, 
#                  weight_decay=0, nomalize_wd=False, eps=1e-8):
#         # if not 0.0 <= lr:
#         #     raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError(f"Invalid epsilon value: {eps}")
#         if not 0.0 <= beta1 < 1.0:
#             raise ValueError(f"Invalid beta1 (momentum) parameter: {beta1}")
#         if not 0.0 <= beta2 < 1.0:
#             raise ValueError(f"Invalid beta2 parameter: {beta2}")
#         defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
#                         eps=eps, weight_decay=weight_decay, nomalize_wd=nomalize_wd)
#         super().__init__(params, defaults)
#         self.eta_min = lr_min / lr
#         self.eta_max = 1.
#         self.eta = 1. # Corresponding to lr_max, while 0. gives lr_min.

#     # def __setstate__(self, state):
#     #     super().__setstate__(state)
#     #     for group in self.param_groups:
#     #         group.setdefault('amsgrad', False)

#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')
#                 # amsgrad = group['amsgrad']

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)
#                     # if amsgrad:
#                     #     # Maintains max of all exp. moving avg. of sq. grad. values
#                     #     state['max_exp_avg_sq'] = torch.zeros_like(p.data)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 # if amsgrad:
#                 #     max_exp_avg_sq = state['max_exp_avg_sq']

#                 # beta1, beta2 = group['betas']
#                 beta1, beta2 = group['beta1'], group['beta2']

#                 state['step'] += 1

#                 # if group['weight_decay'] != 0:
#                 #     grad = grad.add(group['weight_decay'], p.data)

#                 # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 # if amsgrad:
#                 #     # Maintains the maximum of all 2nd moment running avg. till now
#                 #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                 #     # Use the max. for normalizing running avg. of gradient
#                 #     denom = max_exp_avg_sq.sqrt().add_(group['eps'])
#                 # else:
#                 #     denom = exp_avg_sq.sqrt().add_(group['eps'])
#                 denom = exp_avg_sq.sqrt().add_(group['eps'])

#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#                 # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

#                 # weight_decay = group['weight_decay']
#                 # if weight_decay == 0:
#                 #     p.data.addcdiv_(-step_size, exp_avg, denom)
                
#                 step_size = self.eta * group['lr'] * math.sqrt(bias_correction2) / bias_correction1
#                 # lr = self.eta  * alpha #+ (1 - self.eta) * group['lr_min']
#                 # lr is here alpha * eta in the paper (where alpha=gropu['lr'])


#                 weight_decay = group['weight_decay']
#                 if weight_decay != 0:
#                     p.data.mul_(1 - self.eta * weight_decay)

#                 p.data.addcdiv_(-step_size, exp_avg, denom)




#         return loss


# class AdamWR:
#     """Adam with weight decay and warm starts
#     From paper: Fixing weight decay regularization in adam.

#     See fastai: class WeightDecaySchedule(Callback) for normalized weight decay
#     """
#     pass
#     # def __init__(self, params, lr_max, lr_min=1e-5, momentum= 0.9, beta2=0.99, wd=0.,
#     #             normalize_wd=True, eps=1e-8, amsgrad=False):
#     #     betas = (momentum, beta2)
#     #     super().__init__(params, lr_max, betas, eps, 0, amsgrad)


# class SGDR:
#     """Cosine annealing.
#     Warm restarts paper.
#     """
#     pass


# class LrFind:
#     pass


# class OneCycle:
#     """I guess this is linearly changed lr???
#     """

#     pass