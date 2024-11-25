#include "loadmnist.h"
#include <fstream>
#include <iostream>
int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char** readMnistImages(std::string dir, bool train){
    std::string filepath;
    if(train) filepath = dir + "train-images-idx3-ubyte";
    else filepath = dir + "t10k-images-idx3-ubyte";

    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()){
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        if(magic_number != 2051){
            std::cout<<"Error: MNIST Images Magic Number not 2051"<<std::endl;
            return nullptr;
        }
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        unsigned char** images = new unsigned char*[number_of_images];

        for(int i=0;i<number_of_images;++i){
            images[i] = new unsigned char[n_rows * n_cols];
            for(int r=0;r<n_rows;++r){
                for(int c=0;c<n_cols;++c){
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    images[i][r*n_cols + c] = temp;
                }
            }
        }
        file.close();
        return images;
    }
    else{
        std::cout<<"Failed to open MNIST images file: "<<filepath<<std::endl;
        return nullptr;
    }
}

unsigned char* readMnistLabels(std::string dir, bool train){
    std::string filepath;
    if(train) filepath = dir + "train-labels-idx1-ubyte";
    else filepath = dir + "t10k-labels-idx1-ubyte";

    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()){
        int magic_number=0;
        int number_of_labels=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        if(magic_number != 2049){
            std::cout<<"Error: MNIST Labels Magic Number not 2049"<<std::endl;
            return nullptr;
        }
        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
        unsigned char* labels = new unsigned char[number_of_labels];
        for (int i = 0; i < number_of_labels; ++i){
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            labels[i] = temp;
        }
        file.close();
        return labels;
    }
    else{
        std::cout<<"Failed to open MNIST labels file: "<<filepath<<std::endl;
        return nullptr;
    }
}