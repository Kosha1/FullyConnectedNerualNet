CC = g++

default: serialnet

serialnet: main.cpp layer.cpp model.cpp loadmnist.cpp
	${CC} -o $@ -O3 main.cpp layer.cpp model.cpp loadmnist.cpp

clean:
	-rm -f serialnet