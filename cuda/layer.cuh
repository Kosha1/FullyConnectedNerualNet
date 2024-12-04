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
    void gpuForward(T* d_inputs, T* d_outputs, int batch_size, int* d_labels, int* m_correct, bool last = false);

    void printInfo();
    int getNumInputs(){return num_inputs;}
    int getNumOutputs(){return num_outputs;}

    T* getOutputVec(){return output;}
    T* getWeights(){return weights;}

    //prev refers to the layer after this, previous in backpropagation algorithm
    T* calcLayerError(T* prevError, T* prevWeights, int prevNumInputs, int prevNumOutputs);
    T* calcLayerError(int label);//only for last layer use

    float calcLoss(int label);//only for last layer use

    void updateLayerParams(T* weightGrad, T* biasGrad, float learning_rate);

    void updateGPUParams();

private:
    void initRandParams();//randomly initialize weights and biases, only called in constructor

    int num_inputs;
    int num_outputs;
    T* weights;//matrix of dims num_outputs x num_inputs (row major order)
    T* bias;
    T* d_weights;
    T* d_bias;

    //for backpropagation storing (need to store preactivation output and post activation output)
    T* preoutput;//before activation function
    T* output;//after activation function
    T* d_preoutput;
    T* d_output;


    //activation function pointer
    T* (*activation)(T*, T*, int);
};

#endif