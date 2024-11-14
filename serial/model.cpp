#include <iostream>
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
void Model<T>::train(T** trainImages, int imageLength, int count, int* labels, params p){

}

template <typename T>
void Model<T>::backpropagate(T* image, int imageLength, int label){
    //forward pass, each layer class will store intermediate preactivations and activations
    forward(image, imageLength);
    
    //calculate weight and bias gradient for last layer outside of for loop
    // dL/dlastlayer = softmax output - One hot encoding vector
    T* layerError = new T[layers[layers.size() - 1].numOutputs()];
    

    for (int i = layers.size() - 1; i >= 0; --i){

    }
}



template class Model<double>;
template class Model<float>;