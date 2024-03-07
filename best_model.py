import os

import torchtuples as tt


class SaveBestModelCallback(tt.cb.Callback):
    def __init__(self, model_name, dataset_name, net_config, end_epoch, seed, mode=None, start_epoch=None,
                 base_path='best_models'):
        self.best_val_loss = float('inf')
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.mode = mode
        self.net_config = f"{net_config}"  # 将网络配置转换为字符串
        self.filepath = f"{base_path}/{model_name}_{dataset_name}_[{self.net_config}]_{seed}.pth"
        self.best_epoch = None

    def on_epoch_end(self):
        current_epoch = self.model.log.monitors['train_'].epoch

        if self.mode != 'cross_validation':
            val_loss = self.model.log.monitors['val_'].scores['loss']['score'][-1]
            if self.start_epoch <= current_epoch <= self.end_epoch:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = current_epoch
                    print(f"Epoch {current_epoch}: New best model saved with loss: {val_loss} at {self.filepath}")
                    self.model.save_net(self.filepath)

    def load_best_model(self):
        self.model.load_net(self.filepath)
        print(f"Best model loaded from {self.filepath}")
