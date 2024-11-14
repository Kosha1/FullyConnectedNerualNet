CC = g++

default: serialnet

serialnet: main.cpp layer.cpp model.cpp
	${CC} -o $@ main.cpp layer.cpp model.cpp

clean:
	-rm -f serialnet