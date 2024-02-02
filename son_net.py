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

## NOTE: below moved to notebook
# if __name__ == "__main__":
#     ## DEMO: visualizing activation and gradient norms through very deep networks
#     from matplotlib import pyplot as plt

#     # simulate some data
#     BATCH_SIZE = 13
#     INPUT_DIM = DIM

#     data = torch.randn(BATCH_SIZE, INPUT_DIM)

#     # assert we have an orthonormal matrix
#     layer = SOnLinear(DIM, bias=True)
#     assert layer.weights.det().isclose(torch.tensor(1.))
#     layer = SOnLinear(DIM, bias=False)
#     assert layer.weights.det().isclose(torch.tensor(1.))


#     # ensure we've only rotated the input data, not scaled it
#     layer = SOnLinear(DIM, bias=False)
#     output = layer(data)
#     assert output.norm(dim=1).isclose(data.norm(dim=1)).all()

#     layer = SOnLinear(DIM, bias=True)
#     output = layer(data)
#     assert (output - layer.bias).norm(dim=1).isclose(data.norm(dim=1)).all()

#     # lets propagate deep!
#     # init models
#     DEPTH = 1001
#     NONLINEARITY, NLNAME = nn.Tanh(), "tanh"

#     # make two very deep MLPs
#     control, rubiks = [], []
#     for d in range(DEPTH):
#         control += [nn.Linear(DIM, DIM, bias=False), NONLINEARITY]
#         rubiks += [SOnLinear(DIM, bias=False), NONLINEARITY]
#     control = nn.Sequential(*control)
#     rubiks = nn.Sequential(*rubiks)

#     # register hooks for inspection
#     activation = {}
#     grad = {}

#     def get_activation(name):
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook

#     def get_grad(name):
#         def hook(model, grad_input, grad_output):
#             grad[name] = grad_output[0].detach()
#         return hook

#     for d in range(DEPTH):
#         control[d*2].register_forward_hook(get_activation(f"control_{d}"))
#         rubiks[d*2].register_forward_hook(get_activation(f"rubiks_{d}"))

#     for d in range(DEPTH):
#         control[d*2].register_backward_hook(get_grad(f"control_{d}"))
#         rubiks[d*2].register_backward_hook(get_grad(f"rubiks_{d}"))

#     # init optimizers to inspect gradients
#     # forward and backward pass through the network
#     control_out = control(data)
#     rubiks_out = rubiks(data)

#     closs = nn.functional.mse_loss(control_out, data)
#     rloss = nn.functional.mse_loss(rubiks_out, data)

#     closs.backward()
#     rloss.backward()

#     control_norms, rubiks_norms = [], []
#     control_grad_norms, rubiks_grad_norms = [], []
#     for d in range(DEPTH):
#         c = f"control_{d}"
#         r = f"rubiks_{d}"

#         control_norms.append(activation[c].norm(dim=1).mean())
#         rubiks_norms.append(activation[r].norm(dim=1).mean())

#         control_grad_norms.append(grad[c].norm(dim=1).mean())
#         rubiks_grad_norms.append(grad[r].norm(dim=1).mean())

#     # plot our results
#     fig, axes = plt.subplots(1, 2)

#     # activations
#     ax = axes[0]
#     ax.set_title(f'activation norms for {NLNAME}')
#     cl = ax.plot(control_norms, 'b.', label="control")
#     rl = ax.plot(rubiks_norms, 'g.', label="rubiks")
#     ax.legend(title="layer type")
#     ax.set_xscale('log')
#     # ax.set_yscale('log')
#     ax.set_ylabel("mean activation norm")
#     ax.set_xlabel("layer number")

#     # grads
#     ax = axes[1]
#     ax.set_title(f'gradient norms for {NLNAME}')
#     cl = ax.plot(control_grad_norms, 'b.', label="control")
#     rl = ax.plot(rubiks_grad_norms, 'g.', label="rubiks")
#     ax.legend(title="layer type")
#     ax.set_xscale('log')
#     # ax.set_yscale('log')
#     ax.set_ylabel("mean gradient norm")
#     ax.set_xlabel("layer number")

#     plt.show()

#     # activation and gradient norms barely shrink in the SO(n)/Rubik's case!
