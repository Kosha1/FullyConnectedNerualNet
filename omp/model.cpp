#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>
#include "model.h"
#include "layer.h"
#include "functions.h"
#include "util.h"

template <typename T>
Model<T>::Model(T* (*hiddenAct)(T*, T*, int), T* (*outputAct)(T*, T*, int)){
    //num_inputs = 10;
    num_inputs = 784;
    num_outputs = 10;

    //Since Layer object have manual memory management and put into vectors, the c++ rule of 3 must be implemented
    //Layer class has explicit copy constructor and assignment operator implemented
    //push_back invokes copy constructor, if one is not explicitly defined then double free or segfaults will occur
    /*
    layers.push_back(Layer<T>(num_inputs, 10, hiddenAct));
    layers.push_back(Layer<T>(10, 10, hiddenAct));
    //layers.push_back(Layer<T>(10, num_outputs, hiddenAct));
    layers.push_back(Layer<T>(10, num_outputs, outputAct));
    */
    layers.push_back(Layer<T>(num_inputs, 512, hiddenAct));
    layers.push_back(Layer<T>(512, 512, hiddenAct));
    layers.push_back(Layer<T>(512, num_outputs, outputAct));

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

//forward function returns the softmax output of the model
template <typename T>
T* Model<T>::forward(T* input, int size){
    /*
    if (omp_get_thread_num() == 0){
            layers.push_back(Layer<T>(512, 512, nullptr));
        }
        #pragma omp barrier

        #pragma omp critical
        std::cout<<omp_get_thread_num()<<": "<<layers.size()<<std::endl;
    */

    if (size != num_inputs){
        throw std::runtime_error("Input Vector Dimension Incorrect For Model");
    }

    T* prevInput = input;
    int inputsize = size;
    for (int i = 0; i < layers.size(); ++i){
        //std::cout<<"Layer "<<i<<" "<<layers[i].getNumOutputs()<<"x"<<layers[i].getNumInputs()<<std::endl;
        prevInput = layers[i].forward(prevInput, inputsize);
        inputsize = layers[i].getNumOutputs();
    }

    /*
    T sum = 0;
    for(int i = 0; i < inputsize; ++i){
        sum += prevInput[i];
    }
    std::cout<<"Sum: "<<sum<<std::endl;
    */
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
        //std::cout<<"---Epoch "<<e<<"---"<<std::endl;
        //shuffle indices vector, each batch will use batchsize indices from the vector
        std::shuffle(indices.begin(), indices.end(), gen);
        int global_count = 0;
        for(int b = 0; b < num_batches; ++b){//iterate through batches
            bool print_loss = false;
            float batch_loss = 0;
            if (b % 100 == 0){//print loss statistics every 100th batch
                print_loss = true;
                std::cout<<b<<"/"<<num_batches<<" loss: ";
            }

            for(int i = 0; i < p.batch_size && global_count < count; ++i, ++global_count){//iterate through images in one batch
                int image_index = indices[global_count];
                //each image in batch will add its gradients to weightsGrad and biasesGrad arrays
                float loss = backpropagate(trainImages[image_index], imageLength, labels[image_index]);
                //if(print_loss) std::cout<<loss<<" ";
                batch_loss += loss;
                //batch_loss += backpropagate(trainImages[image_index], imageLength, labels[image_index]);
            }
            if (print_loss){
                //std::cout<<std::endl;
                std::cout<<batch_loss<<std::endl;
            }
            //Update the model's weights and biases
            updateModelParams(p.learn_rate);
            //Zero the shared gradients for the next batch
            zeroGrad();
        }
    }
}

template <typename T>
float Model<T>::backpropagate(T* image, int imageLength, int label){
    //forward pass, each layer class will store intermediate preactivations and activations
    forward(image, imageLength);

    //In order to calculate loss: loss = -ln(softmax(output)_label)
    //ln(softmax) naively is numerically unstable; ln(0.000001) --> -infinity
    //ln(softmax) can be calculated in a stable way using the preactivations of the output layer
    float loss = layers[layers.size() - 1].calcLoss(label);

    //std::cout<<"-----Backpropagation Layer Errors-----"<<std::endl;
    
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

    return loss;
}

template <typename T>
void Model<T>::singleBiasGradUpdate(int depth, T* layerError, int biasLength){
    //Bias Gradient = LayerError
    //#pragma omp parallel for
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
    //#pragma omp parallel
    //{
    //#pragma omp for
    for (int i = 0; i < layerLength; ++i){
        for(int j = 0; j < inputLength; ++j){
            singleWeightGrad[i * inputLength + j] = layerError[i] * layerInput[j];
        }
    }
    //+= singleWeightGrad array into shared WeightsGrad
    //#pragma omp for
    for (int i = 0; i < layerLength * inputLength; ++i){
        weightsGrad[depth][i] += singleWeightGrad[i];
    }
    //}
    delete[] singleWeightGrad;
}

template <typename T>
void Model<T>::zeroGrad(){
    for (int i = 0; i < layers.size(); ++i){
        #pragma omp for
        for (int j = 0; j < layers[i].getNumInputs() * layers[i].getNumOutputs(); ++j){
            weightsGrad[i][j] = 0.0;
        }
        #pragma omp for
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

template <typename T>
int Model<T>::test(T** testImages, int imageLength, int count, int* labels){
    int correct = 0;
    //#pragma omp parallel firstprivate(layers) reduction(+: correct)
    //{
        /*   
        if (omp_get_thread_num() == 0){
            layers.push_back(Layer<T>(512, 512, nullptr));
        }
        #pragma omp barrier

        #pragma omp critical
        std::cout<<omp_get_thread_num()<<": "<<layers.size()<<std::endl;
        */
        //#pragma omp for
        for (int i = 0; i < count; ++i){
            T* outputs = forward(testImages[i], imageLength);
            //find index of maximum value
            int max_index = 0;
            T max_val = outputs[0];
            T sum = max_val;
            for (int j = 1; j < num_outputs; ++j){
                sum += outputs[j];
                if (outputs[j] > max_val){
                    max_index = j;
                    max_val = outputs[j];
                }
            }
            /*
            if(sum != 1.0){
                throw std::runtime_error("Testing: Softmax output sum not equal to 1");
            }
            */
            if (max_index == labels[i]){
                //correct++;
                correct += 1;
            }
        }
    //}
    //std::cout<<correct<<"/"<<count<<" correct"<<std::endl;
    return correct;
}

template<typename T>
void Model<T>::addGrad(const Model<T>& l1){
    for (int i = 0; i < layers.size(); ++i){
        for (int j = 0; j < layers[i].getNumInputs() * layers[i].getNumOutputs(); ++j){
            //#pragma omp atomic
            weightsGrad[i][j] += l1.weightsGrad[i][j];
        }
        for (int j = 0; j < layers[i].getNumOutputs(); ++j){
            //#pragma omp atomic
            biasesGrad[i][j] += l1.biasesGrad[i][j];;
        }
    }
}

template <typename T>
Model<T>::Model(const Model<T>& l1):layers(l1.layers){//copy constructor
    num_inputs = l1.num_inputs;
    num_outputs = l1.num_outputs;
    weightsGrad = new T*[layers.size()];
    biasesGrad = new T*[layers.size()];
    for (int i = 0; i < layers.size(); ++i){
        weightsGrad[i] = new T[layers[i].getNumInputs() * layers[i].getNumOutputs()];
        biasesGrad[i] = new T[layers[i].getNumOutputs()];
    }
    //copy values into the arrays
    for (int i = 0; i < layers.size(); ++i){
        for (int j = 0; j < layers[i].getNumInputs() * layers[i].getNumOutputs(); ++j){
            weightsGrad[i][j] = l1.weightsGrad[i][j];
        }
        for (int j = 0; j < layers[i].getNumOutputs(); ++j){
            biasesGrad[i][j] = l1.biasesGrad[i][j];
        }
    }
}

template <typename T>
Model<T>& Model<T>::operator=(const Model<T>& l1){//assignment operator overload
    if (this != &l1){
        for (int i = 0; i < layers.size(); ++i){
            delete[] weightsGrad[i];
            delete[] biasesGrad[i];
        }
        delete[] weightsGrad;
        delete[] biasesGrad;

        layers = l1.layers;//copy assignment of vector
        num_inputs = l1.num_inputs;
        num_outputs = l1.num_outputs;
        weightsGrad = new T*[layers.size()];
        biasesGrad = new T*[layers.size()];
        for (int i = 0; i < layers.size(); ++i){
            weightsGrad[i] = new T[layers[i].getNumInputs() * layers[i].getNumOutputs()];
            biasesGrad[i] = new T[layers[i].getNumOutputs()];
        }
        //copy values into the arrays
        for (int i = 0; i < layers.size(); ++i){
            for (int j = 0; j < layers[i].getNumInputs() * layers[i].getNumOutputs(); ++j){
                weightsGrad[i][j] = l1.weightsGrad[i][j];
            }
            for (int j = 0; j < layers[i].getNumOutputs(); ++j){
                biasesGrad[i][j] = l1.biasesGrad[i][j];
            }
        }
    }
    return *this;
}



template class Model<double>;
template class Model<float>;