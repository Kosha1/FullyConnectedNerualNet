#include <iostream>
#include <vector>
#include <chrono>
#include "layer.cuh"
#include "util.cuh"
#include "functions.h"
#include "model.cuh"
#include "loadmnist.h"
#include "error_check.cuh"

int main(int argc, char** argv){

    if (argc < 2){
        std::cout << "Usage: " << argv[0] << " <GPU Inference Batch Size>" << std::endl;
        return -1;
    }

    const int num_train_images = 60000;
    const int num_test_images = 10000;
    const int image_width = 28;
    const int image_height = 28;

    //load in train and test images and labels
    float** train_images = readMnistImages("../MNIST/", true);
    int* train_labels = readMnistLabels("../MNIST/", true);
    float** test_images = readMnistImages("../MNIST/", false);
    int* test_labels = readMnistLabels("../MNIST/", false);


    //offload testing images and labels to GPU global memory
    //labels are a 1d array, simple 1d malloc and memcpy
    int* d_test_labels;
    CUDA_CHECK(cudaMalloc(&d_test_labels, num_test_images*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_test_labels, test_labels, num_test_images* sizeof(int), cudaMemcpyHostToDevice));
    //1 image is treated as a 1d array and there are 10000 testing images
 
    float* d_test_images;
    CUDA_CHECK(cudaMalloc(&d_test_images, num_test_images*image_height*image_width*sizeof(float)));
    for (int i = 0; i < num_test_images; ++i){
        CUDA_CHECK(cudaMemcpy(d_test_images + i * image_height * image_width, 
                                test_images[i], image_height * image_width * sizeof(float), cudaMemcpyHostToDevice));
    }


    auto relu = ReLU<float>;
    auto softmax = Softmax<float>;


    hyperparams p = {1, 32, 0.01};

    int infer_batch_size = atoi(argv[1]);
    std::cout<<"GPU Inference Batch Size: "<<infer_batch_size<<std::endl;
    Model<float> model = Model<float>(relu, softmax, infer_batch_size);

    for(int i = 0; i < 10; ++i){
        std::cout<<"---Epoch "<<i<<" ---"<<std::endl;
        auto total_start = std::chrono::high_resolution_clock::now();
        //model.test(test_images, image_height*image_width, num_test_images, test_labels);
        int correct = model.gpuInference(d_test_images, d_test_labels, num_test_images, image_height*image_width);
        auto test_end = std::chrono::high_resolution_clock::now();
        std::cout<<correct<<"/"<<num_test_images<<std::endl;
        std::cout<<"Test time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(test_end - total_start).count()<<" ms"<<std::endl;
        model.train(train_images, image_height*image_width, num_train_images, train_labels, p);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::cout<<"Epoch Train time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_end - test_end).count()<<" ms"<<std::endl;
        model.updateGPUParams();
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

    CUDA_CHECK(cudaFree(d_test_labels));
    CUDA_CHECK(cudaFree(d_test_images));

    return 0;
}