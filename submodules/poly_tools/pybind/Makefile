PYTHONARCH=python3
INCLUDE+=$(shell $(PYTHONARCH)-config --includes)
INCLUDE+=-I/usr/include/eigen3 # Eigen3 directory
SUFFIX=$(shell $(PYTHONARCH)-config --extension-suffix)

CC=g++

_poly_tools_cc$(SUFFIX): python_poly_tools.cpp
	$(CC) -O3 -Wall -shared -fPIC -DSCALAR_MODE=0 $(INCLUDE) python_poly_tools.cpp  -o _poly_tools_cc$(SUFFIX)
	mv _poly_tools_cc$(SUFFIX) ../python/poly_tools
