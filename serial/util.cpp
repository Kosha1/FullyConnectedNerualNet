#include <iostream>
#include <random>
#include "util.h"
/*

double* matrixVectorMult(int rows, int cols, double* mat, double* vec){
    double* outVec = new double[rows];

    for (int i = 0; i < rows; ++i){
        double sum = 0.0;
        for (int j = 0; j < cols; ++j){
            sum += mat[i * cols + j] * vec[j];
        }
        outVec[i] = sum;
    }

    return outVec;
}

void vectorAddInPlace(double* vec1, double* vec2, int size){
    for (int i = 0; i < size; ++i){
        vec1[i] = vec1[i] + vec2[i];
    }
}

void printVector(double* vec, int size){
    for (int i = 0; i < size; ++i){
        std::cout<<vec[i]<<" ";
    }
    std::cout<<std::endl;
}
void initRandVector(double* vec, int size, double max, double min){
    //from uniform_real_distribution cpp reference
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(min, max);

    for (int i = 0; i < size; ++i){
        vec[i] = dis(gen);
    }
}
*/