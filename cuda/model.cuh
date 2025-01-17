#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.cuh"


//Training HyperParameters struct
struct hyperparams{
    int epochs;
    int batch_size;
    float learn_rate;
};

template<typename T>
class Model{
public:
    Model(T* (*hiddenAct)(T*, T*, int), T* (*outputAct)(T*, T*, int), int batch_size);
    Model(const Model<T>& l1);//copy constructor
    Model& operator=(const Model<T>& other);//assignment operator overload
    ~Model();

    int gpuInference(T* d_images, int* d_labels, int count, int imageLength);
    T* forward(T* input, int size);

    void updateGPUParams();

    void train(T** trainImages, int imageLength, int count, int* labels, hyperparams p);
    void test(T** testImages, int imageLength, int count, int* labels);

    //calculate derivatives of all layers for one input vector and corresponding label
    float backpropagate(T* image, int imageLength, int label);

private:
    int num_inputs;//input vector dimension to model
    int num_outputs;//final output vector dimension of model
    std::vector<Layer<T>> layers;

    std::vector<T*> d_infer_outputs;
    int infer_batch_size;

    //calculate derivatives of all layers for one input vector and corresponding label
    //void backpropagate(T* image, int imageLength, int label);

    //functions to add one images gradient to shared gradients fields below
    void singleBiasGradUpdate(int depth, T* layerError, int biasLength);
    void singleWeightsGradUpdate(int depth, T* layerError, T* layerInput, int layerLength, int inputLength);

    //Each layer in the model needs a gradient vector for biases and a gradient matrix for weights
    //Every example in a batch will calculate its own gradients and then add them to this shared term
    //at the end, the weights will be updated using this gradient sum
    //different batches will be processed sequentially, even in the parallelized version
    T** weightsGrad;
    T** biasesGrad;

    //before each batch, reset all gradients to 0.0
    void zeroGrad();

    //update all weight and bias parameters in all layers based on weightsGrad and biasesGrad
    void updateModelParams(float learning_rate);
};

#endif