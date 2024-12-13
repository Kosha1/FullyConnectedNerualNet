#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <random>
#include <algorithm>
#include "layer.h"
#include "util.h"
#include "functions.h"
#include "model.h"
#include "loadmnist.h"

int main(int argc, char** argv){
    if (argc < 4){
        std::cout << "Usage: " << argv[0] << " <numthreads> <SGD batch_size> <learning rate>" << std::endl;
        return -1;
    }

    int numthreads = atoi(argv[1]);
    omp_set_num_threads(numthreads);
    std::cout<<"Program running with "<<numthreads<<" threads"<<std::endl;

    const int num_train_images = 60000;
    const int num_test_images = 10000;
    const int image_width = 28;
    const int image_height = 28;

    //load in train and test images and labels
    float** train_images = readMnistImages("../MNIST/", true);
    int* train_labels = readMnistLabels("../MNIST/", true);
    float** test_images = readMnistImages("../MNIST/", false);
    int* test_labels = readMnistLabels("../MNIST/", false);

    auto relu = ReLU<float>;
    auto softmax = Softmax<float>;

    int batch_size = atoi(argv[2]);
    std::cout<<"SGD Batch Size "<<batch_size<<std::endl;
    float learn_rate = atof(argv[3]);
    std::cout<<"Learning rate "<<learn_rate<<std::endl;
    hyperparams p = {1, batch_size, learn_rate};
    //hyperparams p = {1, 512, 0.001};
    //hyperparams p = {1, 64, 0.01};

    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices(num_train_images);
    for (int i = 0; i < num_train_images; ++i){//fill vector from 0 to count-1
        indices[i] = i;
    }
    int num_batches = (num_train_images - 1)/p.batch_size + 1;
    

    Model<float> model = Model<float>(relu, softmax);
    //Model<float> master_model = Model<float>(model);//from copy constructor

    
    //model.train(train_images, image_height*image_width, num_train_images, train_labels, p);
    for(int i = 0; i < 10; ++i){
        std::cout<<"---Epoch "<<i<<" ---"<<std::endl;
        //auto total_start = std::chrono::high_resolution_clock::now();

        std::shuffle(indices.begin(), indices.end(), gen);

        int correct = 0;
        float loss = 0;
        double start_epoch = omp_get_wtime();
        #pragma omp parallel
        {
            Model<float> private_model(model);
            //inference on test images
            double start_infer = omp_get_wtime();
            #pragma omp for reduction(+:correct)
            for(int i = 0; i < num_test_images; ++i){
                correct += private_model.test(test_images+i, image_height*image_width, 1, test_labels+i);
            }
            double stop_infer = omp_get_wtime();
            #pragma omp single nowait
            {
                std::cout<<correct<<"/"<<num_test_images<<" correct"<<std::endl;
                std::cout<<"Inference Time: "<<stop_infer - start_infer<<" s"<<std::endl;
            }
            //training
            for(int i = 0; i < num_batches; ++i){//iterate through the batches in the epoch
                int count = p.batch_size;
                if (i == num_batches - 1){
                    count = num_train_images - (num_batches - 1) * p.batch_size;//last batch may have fewer images than batch_size
                }

                //split images in batch among threads
                #pragma omp for reduction(+: loss)
                for(int j = 0; j < count; ++j){
                    int image_index = indices[i*p.batch_size + j];
                    loss += private_model.backpropagate(train_images[image_index], image_height*image_width, train_labels[image_index]);
                }

                #pragma omp single nowait
                {
                    if (i % 100 == 0){
                        std::cout<<i<<"/"<<num_batches<<": "<<loss<<std::endl;
                    }
                    loss = 0;
                }
                model.addGrad(private_model);
                #pragma omp barrier

                //#pragma omp single
                //{
                    model.updateModelParams(p.learn_rate);
                    model.zeroGrad();
                //}
                private_model = model;
            }
        }
        double end_epoch = omp_get_wtime();
        std::cout<<"Epoch Time: "<<end_epoch - start_epoch<<" s"<<std::endl;
        /*
        auto test_end = std::chrono::high_resolution_clock::now();
        //std::cout<<correct<<"/"<<num_test_images<<" correct"<<std::endl;
        std::cout<<"Test time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(test_end - total_start).count()<<" ms"<<std::endl;
        model.train(train_images, image_height*image_width, num_train_images, train_labels, p);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::cout<<"Epoch Train time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_end - test_end).count()<<" ms"<<std::endl;
        */
    }
    
    /*
    for(int i = 0; i < 10; ++i){
        std::cout<<"---Epoch "<<i<<" ---"<<std::endl;
        auto total_start = std::chrono::high_resolution_clock::now();
        int correct = model.test(test_images, image_height*image_width, num_test_images, test_labels);
        auto test_end = std::chrono::high_resolution_clock::now();
        std::cout<<correct<<"/"<<num_test_images<<std::endl;
        std::cout<<"Test time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(test_end - total_start).count()<<" ms"<<std::endl;
        model.train(train_images, image_height*image_width, num_train_images, train_labels, p);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::cout<<"Epoch Train time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_end - test_end).count()<<" ms"<<std::endl;
    }
    */
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