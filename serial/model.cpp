#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "model.h"
#include "layer.h"
#include "functions.h"
#include "util.h"

template <typename T>
Model<T>::Model(T* (*hiddenAct)(T*, T*, int), T* (*outputAct)(T*, T*, int)){
    num_inputs = 10;;
    num_outputs = 10;

    //Since Layer object have manual memory management and put into vectors, the c++ rule of 3 must be implemented
    //Layer class has explicit copy constructor and assignment operator implemented
    //push_back invokes copy constructor, if one is not explicitly defined then double free or segfaults will occur
    layers.push_back(Layer<T>(num_inputs, 10, hiddenAct));
    layers.push_back(Layer<T>(10, 10, hiddenAct));
    //layers.push_back(Layer<T>(10, num_outputs, hiddenAct));
    layers.push_back(Layer<T>(10, num_outputs, outputAct));

    weightsGrad = new T*[layers.size()];
    biasesGrad = new T*[layers.size()];
    for (int i = 0; i < layers.size(); ++i){
        weightsGrad[i] = new T[layers[i].getNumInputs() * layers[i].getNumOutputs()];
        biasesGrad[i] = new T[layers[i].getNumOutputs()];
    }
    zeroGrad();
}

template <typename T>
Model<T>::~Model(){//should abide by Rule of Three ideally
    for (int i = 0; i < layers.size(); ++i){
        delete[] weightsGrad[i];
        delete[] biasesGrad[i];
    }
    delete[] weightsGrad;
    delete[] biasesGrad;
}

template <typename T>
T* Model<T>::forward(T* input, int size){
    if (size != num_inputs){
        throw std::runtime_error("Input Vector Dimension Incorrect For Model");
    }

    T* prevInput = input;
    int inputsize = size;
    for (int i = 0; i < layers.size(); ++i){
        std::cout<<"Layer "<<i<<" "<<layers[i].getNumOutputs()<<"x"<<layers[i].getNumInputs()<<std::endl;
        prevInput = layers[i].forward(prevInput, inputsize);
        inputsize = layers[i].getNumOutputs();
    }

    T sum = 0;
    for(int i = 0; i < inputsize; ++i){
        sum += prevInput[i];
    }
    std::cout<<"Sum: "<<sum<<std::endl;

    return prevInput;
}

template <typename T>
void Model<T>::train(T** trainImages, int imageLength, int count, int* labels, hyperparams p){
    std::vector<int> indices(count);
    for (int i = 0; i < count; ++i){//fill vector from 0 to count-1
        indices[i] = i;
    }

    int num_batches = (count - 1)/p.batch_size + 1;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int e = 0; e < p.epochs; ++e){//iterate through epochs
        //shuffle indices vector, each batch will use batchsize indices from the vector
        std::shuffle(indices.begin(), indices.end(), gen);
        int global_count = 0;
        for(int b = 0; b < num_batches; ++b){//iterate through batches
            for(int i = 0; i < p.batch_size && global_count < count; ++i, ++global_count){//iterate through images in one batch
                int image_index = indices[global_count];
                //each image in batch will add its gradients to weightsGrad and biasesGrad arrays
                backpropagate(trainImages[image_index], imageLength, labels[image_index]);
            }
            //Update the model's weights and biases
            updateModelParams(p.learn_rate);
            //Zero the shared gradients for the next batch
            zeroGrad();
        }
    }
}

template <typename T>
void Model<T>::backpropagate(T* image, int imageLength, int label){
    //forward pass, each layer class will store intermediate preactivations and activations
    forward(image, imageLength);

    std::cout<<"-----Backpropagation Layer Errors-----"<<std::endl;
    
    //Bias Gradient = LayerError
    //Weights Gradient = LayerError (num_outputs x 1 matrix) * input vector transposed (1 x num_inputs matrix)

    //calculate weight and bias gradients for last layer outside of for loop
    int lastDepth = layers.size() - 1;
    T* prevLayerError = layers[lastDepth].calcLayerError(label);
    singleBiasGradUpdate(lastDepth, prevLayerError, layers[lastDepth].getNumOutputs());
    singleWeightsGradUpdate(lastDepth, prevLayerError, layers[lastDepth - 1].getOutputVec(), layers[lastDepth].getNumOutputs(), layers[lastDepth].getNumInputs());
    
    T* LayerError;//for hidden layers, layer error depends on previous error
    for (int i = layers.size() - 2; i >= 1; --i){
        LayerError = layers[i].calcLayerError(prevLayerError, layers[i+1].getWeights(), layers[i+1].getNumInputs(), layers[i+1].getNumOutputs());
        singleBiasGradUpdate(i, LayerError, layers[i].getNumOutputs());
        singleWeightsGradUpdate(i, LayerError, layers[i-1].getOutputVec(), layers[i].getNumOutputs(), layers[i].getNumInputs());
        //prevLayerError is no longer needed, delete it and assign Layer Error to prevLayerError
        delete[] prevLayerError;
        prevLayerError = LayerError;
    }

    //Very first hidden layer needs to be outside of for loop (since the image is the input)
    LayerError = layers[0].calcLayerError(prevLayerError, layers[1].getWeights(), layers[1].getNumInputs(), layers[1].getNumOutputs());
    singleBiasGradUpdate(0, LayerError, layers[0].getNumOutputs());
    singleWeightsGradUpdate(0, LayerError, image, layers[0].getNumOutputs(), layers[0].getNumInputs());
    
    delete[] prevLayerError;
    delete[] LayerError;
}

template <typename T>
void Model<T>::singleBiasGradUpdate(int depth, T* layerError, int biasLength){
    //Bias Gradient = LayerError
    for (int i = 0; i < biasLength; ++i){
        biasesGrad[depth][i] += layerError[i];
    }
}

template <typename T>
void Model<T>::singleWeightsGradUpdate(int depth, T* layerError, T* layerInput, int layerLength, int inputLength){
    if (layers[depth].getNumOutputs() != layerLength){
        throw std::runtime_error("Shared Gradient Update Output Dimension error");
    }
    if (layers[depth].getNumInputs() != inputLength){
        throw std::runtime_error("Shared Gradient Update Input Dimension error");
    }
    //Weights Gradient = LayerError (num_outputs x 1 matrix) * input vector transposed (1 x num_inputs matrix)
    T* singleWeightGrad = new T [layerLength * inputLength];//matrix dimension matches layer's weight matrix dimension
    for (int i = 0; i < layerLength; ++i){
        for(int j = 0; j < inputLength; ++j){
            singleWeightGrad[i * inputLength + j] = layerError[i] * layerInput[j];
        }
    }
    //+= singleWeightGrad array into shared WeightsGrad
    for (int i = 0; i < layerLength * inputLength; ++i){
        weightsGrad[depth][i] += singleWeightGrad[i];
    }
    delete[] singleWeightGrad;
}

template <typename T>
void Model<T>::zeroGrad(){
    for (int i = 0; i < layers.size(); ++i){
        for (int j = 0; j < layers[i].getNumInputs() * layers[i].getNumOutputs(); ++j){
            weightsGrad[i][j] = 0.0;
        }
        for (int j = 0; j < layers[i].getNumOutputs(); ++j){
            biasesGrad[i][j] = 0.0;
        }
    }
}

template <typename T>
void Model<T>::updateModelParams(float learning_rate){
    for(int l = 0; l < layers.size(); ++l){
        layers[l].updateLayerParams(weightsGrad[l], biasesGrad[l], learning_rate);
    }
}



template class Model<double>;
template class Model<float>;