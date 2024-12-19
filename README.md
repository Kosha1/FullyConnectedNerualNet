# Implementation and Parallelization of a Fully Connected Neural Network From Scratch
## CSEN 145 Parallel Computing Course Final Project

Project contains 3 different implementations of a multi-layer perceptron network.

- Serial Network
  - Written from scratch in C++ with no external frameworks or dependencies
  - Object Oriented design (Model and Layer classes allowing for arbitrary model depth and layer dimensions
  - Implementation of ReLu and Softmax activation functions and cross entropy loss
  - Implementation of backpropagation algorithm for model training
  - Trained and tested model using the MNIST digit dataset
    - Network configured with 784 inputs, 2 hidden layer with 512 neurons each, output layer with 10 neurons
    - Acheives 98% testing accuracy (matching PyTorch testing of the same network)
