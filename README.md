# torchtuples 

**torchtuples** is a small python package for training pytorch models where the data is in the form of nested tuples.
It works equally well for numpy arrays and torch tensors.


## Example

Let us illustrate this with minimal example:
```python
import torch
from torch import nn
from torchtuples import Model, optim
```
Make a data set with three sets of covariates `x0`, `x1` and `x2`, and a target `y`.
The covariates are structure in a neste tuple `x`.
```python
n = 500
x0, x1, x2 = [torch.randn(n, 3) for _ in range(3)]
y = torch.randn(n, 1)
x = (x0, (x0, x1, x2))
```
Create a simple relu net that takes as input the tensor `x_tensor` and the tuple `x_tuple`. Note that `x_tuple` is of arbitrary length. The tensors in `x_tuple` are passed through a layer `lin_tuple`, averaged, and concatenated with `x_tensor`.
We then pass our new tensor though the layer `lin_cat`.
```python
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
```

We can now fit the model with
```python
model = Model(Net(), nn.MSELoss(), optim.SGD(0.01))
log = model.fit(x, y, batch_size=64, epochs=5)
```
and make predictions with
```python
preds = model.predict(x)
```

## Installation

**torchtuples** depends on PyTorch which should be installed from [here](https://pytorch.org/get-started/locally/).
We recommend using python 3.7 as we have not tested the package for any previous versions.

If you are using **conda** we recommend you also install numpy, pandas, and matplotlib.

Next, **torchtuples** can be installed by cloning the repo:
```bash
git clone https://github.com/havakv/torchtuples.git
cd torchtuples
python setup.py install
```
