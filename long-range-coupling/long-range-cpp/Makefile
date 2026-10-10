PYTHONARCH=python3
INCLUDE+=$(shell $(PYTHONARCH)-config --includes)
INCLUDE+=-I/usr/include/eigen3
SUFFIX=$(shell $(PYTHONARCH)-config --extension-suffix)

CC=g++

long_range_cpp$(SUFFIX): long-range.cpp
	$(CC) -O3 -Wall -shared  -fPIC $(INCLUDE) long-range.cpp -o long_range_cpp$(SUFFIX)
	mv long_range_cpp$(SUFFIX) ../