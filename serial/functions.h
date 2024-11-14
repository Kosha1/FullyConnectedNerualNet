#include <math.h>

//Activation of input vector stored in output
template <typename T>
T* ReLU(T* input, T* output, int size){
    for(int i = 0; i < size; ++i){
        if(input[i] < 0.0){
            output[i] = 0.0;
        }
        else{
            output[i] = input[i];
        }
    }
    return output;
}

//Softmax Operation
//FUTURE NOTE: TO AVOID NUMERICAL INSTABILITY:
//Subtract the max value from every element in array
template <typename T>
T* Softmax(T* input, T* output, int size){
    //find max value in input array, will shift everything by this value
    T maxVal = input[0];
    for(int i = 1; i < size; ++i){
        if (input[i] > maxVal) maxVal = input[i];
    }

    T* exponents = new T[size];
    T sum = 0.0;
    for (int i = 0; i < size; ++i){
        exponents[i] = exp(input[i] - maxVal);
        sum += exponents[i];
    }
    for (int i = 0; i < size; ++i){
        output[i] = exponents[i] / sum;
    }
    delete[] exponents;
    return output;
}