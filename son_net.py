import torch
from torch import nn

import numpy as np

class SkewSymmetricMatrix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        self._mat = nn.Parameter(torch.randn(self.n, self.n))

    def forward(self):
        upper = self._mat.triu()
        lower = -upper.t()
        return upper + lower

class OrthogonalMatrix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.skew_symmetric = SkewSymmetricMatrix(self.n)

    def forward(self):
        ss = self.skew_symmetric()
        return ss.matrix_exp()


class SOnLinear(nn.Module):
    """linear layer  where the weight matrix used is guaranteed to be in SO(n)"""
    def __init__(self, n: int, bias: bool=False) -> None:
        super().__init__()
        self.n = n
        self.use_bias = bias

        self.bias = torch.zeros(self.n, requires_grad=self.use_bias)
        self._orth = OrthogonalMatrix(self.n)

    def forward(self, input):
        output = input @ self.weights + self.bias
        return output

    @property
    def weights(self):
        return self._orth()

    @property
    def _skew_symmetric(self):
        return self._orth.skew_symmetric()

if __name__ == "__main__":
    ## DEMO: visualizing activation norms through very deep networks
    from matplotlib import pyplot as plt

    DIM = 11

    # assert we have an orthonormal matrix
    layer = SOnLinear(DIM, bias=True)
    assert layer.weights.det().isclose(torch.tensor(1.))
    layer = SOnLinear(DIM, bias=False)
    assert layer.weights.det().isclose(torch.tensor(1.))

    # simulate some data
    BATCH_SIZE = 13

    data = torch.randn(BATCH_SIZE, DIM)

    # ensure we've only rotated the input data, not scaled it
    layer = SOnLinear(DIM, bias=False)
    output = layer(data)
    assert output.norm(dim=1).isclose(data.norm(dim=1)).all()

    layer = SOnLinear(DIM, bias=True)
    output = layer(data)
    assert (output - layer.bias).norm(dim=1).isclose(data.norm(dim=1)).all()

    # lets propagate deep!
    # init models
    DEPTH = 1001
    # NONLINEARITY, NLNAME = nn.Identity(), "identity"
    NONLINEARITY, NLNAME = nn.Tanh(), "tanh"


    control, rubiks = [], []
    for d in range(DEPTH):
        control += [nn.Linear(DIM, DIM, bias=False), NONLINEARITY]
        rubiks += [SOnLinear(DIM, bias=False), NONLINEARITY]
    control = nn.Sequential(*control)
    rubiks = nn.Sequential(*rubiks)

    # register hooks for inspection
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for d in range(DEPTH):
        control[d*2].register_forward_hook(get_activation(f"control_{d}"))
        rubiks[d*2].register_forward_hook(get_activation(f"rubiks_{d}"))

    # init optimizers to inspect gradients
    control_optim = torch.optim.SGD(
            params = control.parameters(),
            lr=0.1,
        )
    rubiks_optim = torch.optim.SGD(
            params = rubiks.parameters(),
            lr=0.1,
        )

    control_out = control(data)
    rubiks_out = rubiks(data)

    control_norms, rubiks_norms = [], []
    for d in range(DEPTH):
        c = f"control_{d}"
        r = f"rubiks_{d}"
        print(f"control act layer {d}: {activation[c].norm(dim=1).mean()}")
        print(f"rubiks act layer {d}: {activation[r].norm(dim=1).mean()}")

        control_norms.append(activation[c].norm(dim=1).mean())
        rubiks_norms.append(activation[r].norm(dim=1).mean())

    fig, ax = plt.subplots()
    cl = ax.plot(control_norms, 'b-o', label="control")
    rl = ax.plot(rubiks_norms, 'g-o', label="rubiks")
    # leg = ax.legend([cl, rl], ["control", "SO(n)"])
    ax.set_title(f"output norms for non-linearity {NLNAME}")
    ax.legend(title="layer type")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel("mean activation norm")
    ax.set_xlabel("layer number")
    plt.show()

    # # PRINT DATA NORMS after forward pass
    # print(f"depth: {DEPTH}, non-linearity: {NONLINEARITY}")
    # print(f"data norms: {data.norm(dim=1).detach().numpy()}")
    # print(f"control norms: {control_out.norm(dim=1).detach().numpy()}")
    # print(f"rubik norms: {rubiks_out.norm(dim=1).detach().numpy()}")

    loss_fn = nn.MSELoss(reduction="sum")
    target = torch.randn(BATCH_SIZE, DIM)

    control_losses = loss_fn(control_out, target)
    rubiks_losses = loss_fn(rubiks_out, target)
