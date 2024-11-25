#ifndef LOADMNIST_H
#define LOADMNIST_H

#include <string>

//code inspired by https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

int reverseInt(int i);

unsigned char** readMnistImages(std::string dir, bool train=false);

unsigned char* readMnistLabels(std::string dir, bool train=false);

#endif
