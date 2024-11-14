#ifndef LAYER_H
#define LAYER_H

//Template is used to generalize both float and double usage of values in neural network
template <typename T>
class Layer{
public:
    //specify number of input neurons and number of output neurons, with activation function pointer
    Layer(int inputs, int outputs, T* (*actfunc)(T*, T*, int));
    //C++ RULE OF THREE --> needed to avoid segfaults and double frees of memory
    Layer(const Layer<T>& l1);//copy constructor
    Layer& operator=(const Layer<T>& other);//assignment operator overload
    ~Layer();

    T* forward(T* input, int size);

    void printInfo();
    int getNumInputs(){return num_inputs;}
    int getNumOutputs(){return num_outputs;}

    //prev refers to the layer after this, previous in backpropagation algorithm
    T* calcLayerError(T* prevError, T* prevWeights, int prevNumInputs, int prevNumOutputs);
    T* calcLayerError(int label);//only for last layer use

private:
    void initRandParams();//randomly initialize weights and biases, only called in constructor

    int num_inputs;
    int num_outputs;
    T* weights;//matrix of dims num_outputs x num_inputs (row major order)
    T* bias;

    //for backpropagation storing (need to store preactivation output and post activation output)
    T* preoutput;//before activation function
    T* output;//after activation function


    //activation function pointer
    T* (*activation)(T*, T*, int);
};

#endif