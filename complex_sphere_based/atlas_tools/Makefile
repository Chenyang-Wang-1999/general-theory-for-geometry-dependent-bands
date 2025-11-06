PYTHONARCH=python3
# PYTHONARCH=/bin/python3
INCLUDE+=$(shell $(PYTHONARCH)-config --includes)
SUFFIX=$(shell $(PYTHONARCH)-config --extension-suffix)

CC=g++
test_module$(SUFFIX): python_types.cpp
	$(CC) -shared -fPIC -O3 python_types.cpp $(INCLUDE) -o test_module$(SUFFIX)
