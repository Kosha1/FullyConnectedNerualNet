#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.h"


//Training Parameters struct
struct params{
    int epochs;
    int batch_size;
    int learn_rate;
};

template<typename T>
class Model{
public:
    Model(T* (*hiddenAct)(T*, T*, int), T* (*outputAct)(T*, T*, int));
    ~Model();

    T* forward(T* input, int size);

    void train(T** trainImages, int imageLength, int count, int* labels, params p);

private:
    int num_inputs;//input vector dimension to model
    int num_outputs;//final output vector dimension of model
    std::vector<Layer<T>> layers;

    //calculate derivatives of all layers for one input vector and corresponding label
    void backpropagate(T* image, int imageLength, int label);


    //Each layer in the model needs a gradient vector for biases and a gradient matrix for weights
    //Every example in a batch will calculate its own gradients and then add them to this shared term
    //at the end, the weights will be updated using this gradient sum
    //different batches will be processed sequentially, even in the parallelized version
    T** weightsGrad;
    T** biasesGrad;
};

#endif