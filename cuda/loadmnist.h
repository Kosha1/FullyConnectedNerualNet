#ifndef LOADMNIST_H
#define LOADMNIST_H

#include <string>

//code inspired by https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

int reverseInt(int i);

float** readMnistImages(std::string dir, bool train=false);

int* readMnistLabels(std::string dir, bool train=false);

#endif
