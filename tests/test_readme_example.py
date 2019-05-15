
def test_readme_example():
    import torch
    from torch import nn
    from torchtuples import Model, optim
    torch.manual_seed(0)

    n = 500
    x0, x1, x2 = [torch.randn(n, 3) for _ in range(3)]
    y = torch.randn(n, 1)
    x = (x0, (x0, x1, x2))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_tuple = nn.Linear(3, 2)
            self.lin_cat = nn.Linear(5, 1)
            self.relu = nn.ReLU()

        def forward(self, x_tensor, x_tuple):
            x = [self.relu(self.lin_tuple(xi)) for xi in x_tuple]
            x = torch.stack(x).mean(0)
            x = torch.cat([x, x_tensor], dim=1)
            return self.lin_cat(x)
    
    model = Model(Net(), nn.MSELoss(), optim.SGD(0.01))
    log = model.fit(x, y, batch_size=64, epochs=5, verbose=False)
    preds = model.predict(x)
    assert preds is not None
