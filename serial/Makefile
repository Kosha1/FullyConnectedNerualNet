CC = g++

default: serialnet

serialnet: main.cpp layer.cpp model.cpp loadmnist.cpp
	${CC} -o $@ main.cpp layer.cpp model.cpp loadmnist.cpp

clean:
	-rm -f serialnet