PYTHONARCH=python3
INCLUDE+=$(shell $(PYTHONARCH)-config --includes)
INCLUDE+=-I/usr/include/eigen3
SUFFIX=$(shell $(PYTHONARCH)-config --extension-suffix)

CC = g++

lib_GBZ_manifold${SUFFIX}: py_GBZ_manifold.cpp
	${CC} -shared -fPIC -O3 py_GBZ_manifold.cpp ${INCLUDE} -o lib_GBZ_manifold${SUFFIX}
	mv lib_GBZ_manifold${SUFFIX} GBZ_manifold