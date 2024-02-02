# SOn_net/Rubik's Net
## About
A network with linear layers constrained to SO(n) (invertible matrices with determinate 1).  This constraint leaves the network highly flexible yet it's contraint means activations of inputs passed in are only high dimensional rotations of the input (hense also calling it Rubik's net).  This keeps the activation at the same norm as the input, meaning we can pass through many layers without exploding or vanishing the activation (and so also the back-propagated gradients), allowing stable training with many more layers.

## Installation
`pip install torch torchvision matplotlib numpy`.  The network is written in vanilla pytorch.

## Usage
See demo notebook for usage and performance comparisions when working with very deep networks
