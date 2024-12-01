CC = g++

default: ompnet

ompnet: main.cpp layer.cpp model.cpp loadmnist.cpp
	${CC} -o $@ -fopenmp -O3 main.cpp layer.cpp model.cpp loadmnist.cpp

clean:
	-rm -f ompnet