#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <random>

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

#endif