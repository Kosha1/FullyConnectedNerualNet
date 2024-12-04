#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <random>
#include <cmath>

template <typename T>
void matrixVectorMult(int rows, int cols, T* mat, T* vec, T* outVec){
    for (int i = 0; i < rows; ++i){
        T sum = 0.0;
        for (int j = 0; j < cols; ++j){
            sum += mat[i * cols + j] * vec[j];
        }
        outVec[i] = sum;
    }
}

//Adds vec1 and vec2 and stores result in result
template <typename T>
void vectorAdd(T* vec1, T* vec2, T* result, int size){
    for (int i = 0; i < size; ++i){
        result[i] = vec1[i] + vec2[i];
    }
}

//Adds vec1 and vec2 and stores result in vec1
template <typename T>
void vectorAddInPlace(T* vec1, T* vec2, int size){
    for (int i = 0; i < size; ++i){
        vec1[i] = vec1[i] + vec2[i];
    }
}

template <typename T>
void printVector(T* vec, int size){
    for (int i = 0; i < size; ++i){
        std::cout<<vec[i]<<" ";
    }
    std::cout<<std::endl;
}

template <typename T>
void initRandVector(T* vec, int size, T max = 2.0, T min = -2.0){
    //from uniform_real_distribution cpp reference
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(min, max);

    for (int i = 0; i < size; ++i){
        vec[i] = dis(gen);
    }
}

template <typename T>
void HeInitialization(T* vec, int size, int num_inputs){
    T stddev = sqrt(2.0/num_inputs);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<T> d{0.0, stddev};
    for (int i = 0; i < size; ++i){
        vec[i] = d(gen);
    }
}

template <typename T>
__global__ void ReluLayer(T* d_weights, T* d_bias, int num_inputs, int num_outputs, T* d_input, T* d_output){

    int image_num = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //each thread works on one element in the output vector
    if (tid < num_outputs){
        T sum = 0;
        for (int i = 0; i < num_inputs; ++i){//Matrix vector multiplication
            sum += d_weights[tid*num_inputs + i] * d_input[num_inputs*image_num + i];
        }
        //ReLu: max(0.0, preoutput)
        T output = fmaxf(0.0, sum + d_bias[tid]);//bias addition followed by ReLu

        d_output[num_outputs*image_num + tid] = output;
    }
}

template <typename T>
__global__ void PreActLayer(T* d_weights, T* d_bias, int num_inputs, int num_outputs, T* d_input, T* d_output){
    int image_num = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_outputs){
        T sum = 0;
        for (int i = 0; i < num_inputs; ++i){//Matrix vector multiplication
            sum += d_weights[tid*num_inputs + i] * d_input[num_inputs*image_num + i];
        }
        T output = sum + d_bias[tid];
        d_output[num_outputs*image_num + tid] = output;
    }
}

template <typename T>
__global__ void SoftmaxLabels(T* d_inputs, int* labels, int* correct){
    int tid = threadIdx.x;

    T max = -INFINITY;
    int max_index = 0;
    int offset = 10 * tid;
    for (int i = 0; i < 10; ++i){
        if (d_inputs[offset+i] > max){
            max = d_inputs[offset+i];
            max_index = i;
        }
    }
    if(labels[tid] == max_index) atomicAdd(correct, 1);
}

#endif