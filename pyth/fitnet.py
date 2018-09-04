import torch
from torch import nn
from pycox.callbacks import callbacks as cb
from pycox.models.base import BaseModel

class FitNet(BaseModel):
    def __init__(self, net, loss_func, optimizer=None, device=None):
        super().__init__(net, optimizer, device)
        self.loss_func = loss_func

    #@staticmethod
    #def make_dataloader(X, y, batch_size, num_workers):
    #    trainset = PrepareData(X, y)
    #    dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True,
    #                                 num_workers=num_workers)
    #    return dataloader

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=1):
        self._setup_train_info(dataloader, verbose, callbacks)
        self.callbacks.on_fit_start()
        for _ in range(epochs):
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.net(X)
                self.batch_loss = self.loss_func(out, y)
                self.optimizer.zero_grad()
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
        
    
    #def fit_numpy(self, X, y, batch_size=64, epochs=1, num_workers=0, callbacks=None, verbose=1):
    #    dataloader = self.make_dataloader(X, y, batch_size, num_workers)
    #    return self.fit_dataloader(dataloader, epochs, callbacks, verbose)

    def predict_dataloader(self, dataloader, return_numpy=True, eval_=True):
        return self._predict_func_dataloader(self.net, dataloader, return_numpy, eval_)
 
    def predict_numpy(self, X, batch_size=8224, return_numpy=True, eval_=True):
        return self._predict_func_numpy(self.net, X, batch_size, return_numpy, eval_)
    
    def fit(self):
        raise NotImplemented
        
    def predict(self):
        raise NotImplemented
    
    
def dataloader_x(dataloader):
    for x, y in dataloader:
        yield x

class MonitorXyDL(cb.MonitorBase):
    def __init__(self, dataloader, monitor_funcs, per_epoch=1):
        super().__init__(monitor_funcs, per_epoch, batch_size=None)
        self.dataloader = dataloader
        self.y_true = self.get_y()
    
    def get_y(self):
        return torch.cat([y for _, y in self.dataloader])

    def get_score_args(self):
        preds = self.model.predict_dataloader(dataloader_x(self.dataloader), return_numpy=False, eval_=True)
        return preds, self.y_true
 

