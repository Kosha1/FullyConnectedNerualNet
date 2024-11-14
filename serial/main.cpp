#include <iostream>
#include <vector>
#include "layer.h"
#include "util.h"
#include "functions.h"
#include "model.h"

int main(){

    //double* arr = new double[10];
    float* arr = new float[10];
    const int arrsize = 10;
    initRandVector(arr, arrsize);
    //printVector(arr, arrsize);
    
    //auto relu = ReLU<double>;
    //auto softmax = Softmax<double>;
    auto relu = ReLU<float>;
    auto softmax = Softmax<float>;
    //Layer h1 = Layer<float>(10, 10, relu);


    //h1.forward(arr);
    //float* arr1 = h1.forward(arr);
    //delete [] arr1;
    //h1.printInfo();
    //h1.forward();

    //Model<double> model = Model<double>(relu, softmax);
    Model<float> model = Model<float>(relu, softmax);
    model.forward(arr, arrsize);

    delete[] arr;

    return 0;
}