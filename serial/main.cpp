#include <iostream>
#include <vector>
#include "layer.h"
#include "util.h"
#include "functions.h"
#include "model.h"
#include "loadmnist.h"

int main(){


    const int num_train_images = 60000;
    const int num_test_images = 10000;
    const int image_width = 28;
    const int image_height = 28;

    //load in train and test images and labels
    unsigned char** train_images = readMnistImages("../MNIST/", true);
    unsigned char* train_labels = readMnistLabels("../MNIST/", true);
    unsigned char** test_images = readMnistImages("../MNIST/", false);
    unsigned char* test_labels = readMnistLabels("../MNIST/", false);


    //print out one image
    int index = 456;
    for(int i = 0; i < image_height; ++i){
        for(int j = 0; j < image_width; ++j){
            if(test_images[index][i*image_width + j] > 127) std::cout<<1;
            else std::cout<<0;
        }
        std::cout<<std::endl;
    }

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
    //model.forward(arr, arrsize);
    model.backpropagate(arr, arrsize, 3);

    delete[] arr;


    //delete test and train images
    for (int i = 0; i < num_train_images; ++i){
        delete[] train_images[i];
    }
    for(int i = 0; i < num_test_images; ++i){
        delete[] test_images[i];
    }
    delete[] train_images;
    delete[] test_images;
    delete[] train_labels;
    delete[] test_labels;

    return 0;
}