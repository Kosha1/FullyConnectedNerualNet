#include <iostream>
#include "layer.h"
#include "util.h"

template <typename T>
Layer<T>::Layer(int inputs, int outputs, T* (*actfunc)(T*, T*, int)){
    num_inputs = inputs;
    num_outputs = outputs;
    //weights is a num_outputs x num_inputs matrix
    weights = new T [num_inputs * num_outputs];

    //bias vector has num_outputs elements
    bias = new T [num_outputs];

    //initialize the stored output arrays
    preoutput = new T [num_outputs];
    output = new T [num_outputs];

    //init biases and weights
    initRandParams();

    //printVector(bias, num_outputs);

    //store the activation function pointer
    activation = actfunc;
}

template <typename T>
void Layer<T>::printInfo(){
    std::cout<<"Inputs: "<<num_inputs<<std::endl;
    std::cout<<"Outputs: "<<num_outputs<<std::endl;
    std::cout<<"Number of Parameters: "<<num_inputs * num_outputs + num_outputs<<std::endl;
}

template <typename T>
T* Layer<T>::forward(T* input, int size){
    if (num_inputs != size){
        throw std::runtime_error("Layer Input Vector Dimension Wrong");
    }

    //Preactivation: Ax + b, store in preoutput
    matrixVectorMult(num_outputs, num_inputs, weights, input, preoutput);//Ax-->preoutput
    vectorAddInPlace(preoutput, bias, num_outputs);//preoutput + bias -->preoutput
    std::cout<<"   Preact: ";
    printVector(preoutput, num_outputs);

    //apply activation function to preoutput vector, store in output
    activation(preoutput, output, num_outputs);
    std::cout<<"   Postact: ";
    printVector(output, num_outputs);

    return output;
}

template <typename T>
T* Layer<T>::calcLayerError(T* prevError, T* prevWeights, int prevNumInputs, int prevNumOutputs){//all hidden layers
    //prevError must be of size prevNumOutputs, prevNumInputs must equal to this.num_ouputs
    if (prevNumInputs != num_outputs){
        throw std::runtime_error("Backpropagation Layer Dimension errors");
    }

    // dL/dlayer = dh/df * dfprev/dh * dL/dprevLayer
    // dh/df is derivate of activation output with respect to preactivation output (ReLu in our case)
    T* activationDer = new T[num_outputs];
    for (int i = 0; i < num_outputs; ++i){//ReLu derivative: 0 if preact < 0, 1 if preact is >= 0
        if (output[i] < 0) activationDer[i] = 0.0;
        else activationDer[i] = 1.0;
    }

    // dfprev/dh = prevWeight transposed (prevWeights is a prevNumOutputs x prevNumInputs matrix)
    //store prevWeights transposed * prevError in layerError, then pointwise multiply it with activationDer
    T* layerError = new T[num_outputs];
    for (int i = 0; i < prevNumInputs; ++i){
        T sum = 0.0;
        for(int j = 0; j < prevNumOutputs; ++j){
            //row j column i in prevWeights matrix
            sum += prevWeights[j * prevNumInputs + i] * prevError[j];
        }
        layerError[i] = sum;
    }
    //pointwise mult layerError with activation Der
    for(int i = 0; i < num_outputs; ++i){
        layerError[i] = layerError[i] * activationDer[i];
    }
    delete[] activationDer;

    printVector(layerError, num_outputs);
    return layerError;
}

template <typename T>
T* Layer<T>::calcLayerError(int label){//last layer error based on cross entropy loss
    // dL/dlastlayer = softmax output - one hot vector
    T* layerError = new T[num_outputs];
    for (int i = 0; i < num_outputs; ++i){
        layerError[i] = output[i];
    }
    layerError[label] = layerError[label] - 1.0;//subtraction of one hot vector from softmax

    printVector(layerError, num_outputs);

    return layerError;
}

template <typename T>
void Layer<T>::updateLayerParams(T* weightGrad, T* biasGrad, float learning_rate){
    //update weight matrix
    for(int i = 0; i < num_inputs * num_outputs; ++i){
        weights[i] -= weightGrad[i] * learning_rate;
    }
    //update bias vector
    for(int i = 0; i < num_outputs; ++i){
        bias[i] -= biasGrad[i] * learning_rate;
    }
}

template <typename T>
void Layer<T>::initRandParams(){
    T minVal = -2.0;
    T maxVal = 2.0;
    //random weights init
    initRandVector(weights, num_inputs*num_outputs, minVal, maxVal);
    //random bias init
    initRandVector(bias, num_outputs, minVal, maxVal);
    
}

template <typename T>
Layer<T>::Layer(const Layer<T>& l1){//copy constructor
    num_inputs = l1.num_inputs;
    num_outputs = l1.num_outputs;
    weights = new T [num_inputs * num_outputs];

    bias = new T [num_outputs];
    preoutput = new T[num_outputs];
    output = new T[num_outputs];

    //copy weights from l1 in this
    for (int i = 0; i < num_outputs; ++i){
        bias[i] = l1.bias[i];
        preoutput[i] = l1.preoutput[i];
        output[i] = l1.output[i];
    }
    for (int i = 0; i < num_inputs * num_outputs; ++i){
        weights[i] = l1.weights[i];
    }

    activation = l1.activation;
}

template <typename T>
Layer<T>& Layer<T>::operator=(const Layer<T>& l1){
    if (this != &l1){
        delete[] weights;
        delete[] bias;
        num_inputs = l1.num_inputs;
        num_outputs = l1.num_outputs;
        weights = new T [num_inputs * num_outputs];

        bias = new T [num_outputs];
        preoutput = new T[num_outputs];
        output = new T[num_outputs];

        //copy weights from l1 in this
        for (int i = 0; i < num_outputs; ++i){
            bias[i] = l1.bias[i];
            preoutput[i] = l1.preoutput[i];
            output[i] = l1.output[i];
        }
        for (int i = 0; i < num_inputs * num_outputs; ++i){
            weights[i] = l1.weights[i];
        }

        activation = l1.activation;
    }
    return *this;
}

template <typename T>
Layer<T>::~Layer(){
    delete [] weights;
    delete [] bias;
    delete [] preoutput;
    delete [] output;
}

//Template classes can not be simply implemented in a cpp file, otherwise linker error
//Must declare the instances of the template class you will use. Alternative: implement everything in header file
template class Layer<double>;
template class Layer<float>;