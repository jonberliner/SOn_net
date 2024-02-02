# SOn_net/Rubik's Net
## About
A network with linear layers constrained to SO(n) (invertible matrices with determinate 1).  This constraint leaves the network highly flexible yet it's contraint means activations of inputs passed in are only high dimensional rotations of the input (hense also calling it Rubik's net).  This keeps the activation at the same norm as the input, meaning we can pass through many layers without exploding or vanishing the activation (and so also the back-propagated gradients), allowing stable training with many more layers.

## Installation
simply run `pip install torch matplotlib`.  The network is written in vanilla pytorch.

## Usage
run `python son_net.py`.  It will output graphs comparing the norms of activations and gradients of a 1001 layer deep neural network with tanh activations  standard MLP and one for one with Rubik's layers.
