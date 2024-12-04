
default: cudanet

cudanet: main.cu layer.cu model.cu loadmnist.cpp
	nvcc -o $@ -O3 -arch=sm_61 main.cu layer.cu model.cu loadmnist.cpp

clean:
	-rm -f cudanet