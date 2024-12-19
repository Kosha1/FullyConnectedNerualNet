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
    - Acheives 98% testing accuracy (matching PyTorch testing of the same network  structure)
- OpenMP Network
  - CPU parallelized version of neural network using OpenMP
  - Stochastic Gradient Descent Training algorithm is parallelized using mutliple CPU cores
  - Inference of the 10,000 MNIST testing images is also parallelized
- CUDA Network
  - GPU parallelized version of neural network using CUDA
  - Parallelization of inference of MNIST testing images using CUDA kernels

MNIST dataset is not included in the repository. In the root directory make a folder called MNIST.
All 4 files of the MNIST digit dataset must be placed in this folder (testing images, testing labels, training images, training labels) with the default file names as found in the [MNIST specification](https://yann.lecun.com/exdb/mnist/) (files must not be gzipped).

Makefiles are provided for each of the 3 networks. Serial and OpenMP networks require GCC and CUDA network requires NVCC. (CUDA architecture in Makefile configured to be sm_61 for GTX 1080).

Running the programs:
- Serial: `./serialnet`
- OpenMP: `./ompnet <num threads> <SGD batch size> <learning rate>`
  - Warning: learning rate is dependent on batch size
    - If improper combinations are selected, the model will diverge and the loss will output nan
    - Working combination: bs=32 and lr=0.01. If batch size is increased, decrease the learning rate by the same amount.
- CUDA: `./cudanet <GPU Inference Batch Size>`

