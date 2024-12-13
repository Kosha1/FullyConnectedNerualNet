#include <iostream>
#include <vector>
#include <chrono>
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
    float** train_images = readMnistImages("../MNIST/", true);
    int* train_labels = readMnistLabels("../MNIST/", true);
    float** test_images = readMnistImages("../MNIST/", false);
    int* test_labels = readMnistLabels("../MNIST/", false);


    /*
    //print out one image
    int index = 0;
    for(int i = 0; i < image_height; ++i){
        for(int j = 0; j < image_width; ++j){
            if(test_images[index][i*image_width + j] >= 0.5) std::cout<<1;
            else std::cout<<0;
        }
        std::cout<<std::endl;
    }
    std::cout<<"Label: "<<test_labels[index]<<std::endl;
    */

    
    auto relu = ReLU<float>;
    auto softmax = Softmax<float>;


    //hyperparams p = {10, 32, 0.001};
    hyperparams p = {1, 32, 0.01};

    //Model<double> model = Model<double>(relu, softmax);
    Model<float> model = Model<float>(relu, softmax);

    for(int i = 0; i < 10; ++i){
        std::cout<<"---Epoch "<<i<<" ---"<<std::endl;
        auto total_start = std::chrono::high_resolution_clock::now();
        model.test(test_images, image_height*image_width, num_test_images, test_labels);
        auto test_end = std::chrono::high_resolution_clock::now();
        std::cout<<"Test time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(test_end - total_start).count()<<" ms"<<std::endl;
        model.train(train_images, image_height*image_width, num_train_images, train_labels, p);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::cout<<"Epoch Train time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_end - test_end).count()<<" ms"<<std::endl;
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    model.test(test_images, image_height*image_width, num_test_images, test_labels);
    auto test_end = std::chrono::high_resolution_clock::now();
    std::cout<<"Test time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(test_end - total_start).count()<<" ms"<<std::endl;


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