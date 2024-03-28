This directory contains material for high-level (simpler) usage of neural networks.

The `regression` directory contains the following:
- An implementation of a multi-layer neural network (in `neural-net.kojo`) for regression, which can be used in a very simple manner.
  - The number of hidden layers and the number of units in these layers in the network can be specified when the neural network is constructed.
  - The network assumes 1 input and 1 output.
  - The hidden layers use the relu activation function.
  - MSE loss is used.
- Many examples that use the above neural network.

The `classification` directory contains the following:
- An implementation of a multi-layer neural network (in `classification-net.kojo`) for classification, which can be used in a very simple manner.
  - The number of layers and the number of units in these layers can be specified when the neural network is constructed.
  - The hidden layers use the relu activation function.
  - The output layer uses the softmax activation function.
  - Cross-entropy loss is used.
- An MNIST (digit recognition) example. This includes:
  - A training script.
  - A batch mode testing script.
  - An interactive testing script.
