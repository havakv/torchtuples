import torchtuples as tt


class ReduceLROnPlateauCallback(tt.cb.Callback):
    def __init__(self, optimizer, factor=0.1, patience=20, min_lr=1e-6):
        super().__init__()
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience  # Number of epochs to wait for improvement before reducing the learning rate
        self.min_lr = min_lr
        self.best_loss = float('inf')  # Initialize to infinity
        self.wait = 0  # Number of epochs since the last improvement in loss

    def on_epoch_end(self):
        # Get the current valid loss
        current_loss = self.model.log.to_pandas().val_loss.iloc[-1]

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            # wait reaches patience
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    # Calculate the new learning rate and ensure it's not lower than the minimum learning rate
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    # update lr and reset the wait
                    if new_lr < param_group['lr']:
                        param_group['lr'] = new_lr
                        self.wait = 0
                        print(f"Reducing learning rate to {new_lr}")


class AdjustLRCallback(tt.cb.Callback):
    def __init__(self, optimizer, step_size=20, gamma=0.9):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch_count = 0

    def on_epoch_end(self):
        self.epoch_count += 1
        # check if the epoch count is a multiple of the step size
        if self.epoch_count % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
                print(f"Learning rate adjusted to {param_group['lr']}")
