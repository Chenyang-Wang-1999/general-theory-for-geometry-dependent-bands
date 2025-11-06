/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-09-04
 * @lastModified  2023-09-04
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "atlas/GBZ_manifolds.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py=pybind11;

PYBIND11_MAKE_OPAQUE(Atlas::ComplexVec);

PYBIND11_MODULE(test_winding, m)
{
    py::bind_vector<Atlas::ComplexVec>(m, "CComplexVec");
    m.def("bubble_gen", &Atlas::bubble_gen);
    m.def("long_bubble_gen", &Atlas::long_bubble_gen);
}
