from torch import optim
# from .callbacks import Callback



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


class AdamWR:
    """Adam with weight decay and warm starts
    From paper: Fixing weight decay regularization in adam.

    See fastai: class WeightDecaySchedule(Callback) for normalized weight decay
    """
    pass
    # def __init__(self, params, lr_max, lr_min=1e-5, momentum= 0.9, beta2=0.99, wd=0.,
    #             normalize_wd=True, eps=1e-8, amsgrad=False):
    #     betas = (momentum, beta2)
    #     super().__init__(params, lr_max, betas, eps, 0, amsgrad)


class SGDR:
    """Cosine annealing.
    Warm restarts paper.
    """
    pass


class LrFind:
    pass


class OneCycle:
    """I guess this is linearly changed lr???
    """

    pass