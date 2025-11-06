PYTHONARCH=python3
# PYTHONARCH=/bin/python3
INCLUDE+=$(shell $(PYTHONARCH)-config --includes)
INCLUDE+=-I/usr/include/eigen3
SUFFIX=$(shell $(PYTHONARCH)-config --extension-suffix)

CC = g++

test_toy_model$(SUFFIX): toy_model.cpp
	$(CC) -O3 -shared -fPIC toy_model.cpp $(INCLUDE) -o test_toy_model$(SUFFIX)

c_aGBZ_1D_solver$(SUFFIX): python_aGBZ_1d.cpp
	$(CC) -Wall -O3 -shared -fPIC $(INCLUDE)  python_aGBZ_1d.cpp -o c_aGBZ_1D_solver$(SUFFIX)

c_aGBZ_2D_solver$(SUFFIX): python_aGBZ_2d.cpp
	$(CC) -Wall -O3 -shared -fPIC $(INCLUDE)  python_aGBZ_2d.cpp -o c_aGBZ_2D_solver$(SUFFIX)
	mv c_aGBZ_2D_solver$(SUFFIX) c_solver_2D

test_GBZ_manifolds.exe: test_GBZ_manifolds.cpp
	$(CC) -Wall -O3 test_GBZ_manifolds.cpp -o test_GBZ_manifolds.exe

test_polynomial_evaluations.exe: test_polynomial_evaluations.cpp
	$(CC) -Wall -O3 test_polynomial_evaluations.cpp -o test_polynomial_evaluations.exe

test_GBZ_checker_1D.exe: test_GBZ_checker_1D.cpp
	$(CC) -Wall -O3 test_GBZ_checker_1D.cpp -o test_GBZ_checker_1D.exe

test_winding.exe: test_winding.cpp
	$(CC) -Wall -O3 test_winding.cpp -o test_winding.exe

test_winding$(SUFFIX): python_test_winding.cpp
	$(CC) -Wall -O3 -shared -fPIC $(INCLUDE) python_test_winding.cpp -o test_winding$(SUFFIX)

test_basics.exe: test_basics.cpp
	$(CC) -Wall -O3 test_basics.cpp -o test_basics.exe