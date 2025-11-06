/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-25
 * @lastModified  2023-08-17
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "aGBZ_1d.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

namespace py=pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<aGBZ1DSolver::DataPointViewType::ScalarType>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<Atlas::ChartIdType>);

PYBIND11_MODULE(c_aGBZ_1D_solver, m){
    py::bind_vector<std::vector<aGBZ1DSolver::DataPointViewType::ScalarType>>(m, "ScalarVec");
    py::bind_vector<std::vector<int>>(m, "IntVec");
    py::bind_vector<std::vector<std::string>>(m, "StrVec");
    py::bind_vector<std::vector<Atlas::ChartIdType>>(m, "ChartIdVec");
    py::class_<aGBZ1DSolver>(m, "aGBZ1DSolver")
        .def(py::init<Atlas::RealType, Atlas::IndexType, Atlas::IndexType, Atlas::RealType>())
        .def("set_param", &aGBZ1DSolver::set_param)
        .def("set_char_poly_by_coeffs", &aGBZ1DSolver::set_char_poly_by_coeffs)
        .def("next", &aGBZ1DSolver::next)
        .def("continuation_solver_fun", &aGBZ1DSolver::continuation_solver_fun)
        .def_readonly("curr_initial_values", &aGBZ1DSolver::curr_initial_values)
        .def("insert_continuation_solver_results", &aGBZ1DSolver::insert_continuation_solver_results)
        .def("insert_full_solver_results", &aGBZ1DSolver::insert_full_solver_results)
        .def("suspend_solver", &aGBZ1DSolver::suspend_solver)
        .def("post_process", &aGBZ1DSolver::post_process)
        .def("loop_readout", &aGBZ1DSolver::loop_readout)
        .def("N_loops", [](aGBZ1DSolver & solver){return solver.all_loops.size();})
        .def("N_segments", [](aGBZ1DSolver & solver){return solver.all_segments.size();})
        .def("segment_readout", &aGBZ1DSolver::segment_readout)
        .def("raw_sol_readout", &aGBZ1DSolver::raw_sol_readout)
        .def("N_points", [](aGBZ1DSolver & solver){return solver.all_points.size();})
        .def("sol_to_segments", &aGBZ1DSolver::sol_to_segments)
        .def("segments_to_loops", &aGBZ1DSolver::segments_to_loops)
        .def("get_curr_param", &aGBZ1DSolver::get_curr_param)
        .def_readonly("solver_suspended", &aGBZ1DSolver::solver_suspended)
        .def_readonly("full_solver_flag", &aGBZ1DSolver::full_solver_flag)
        .def_readonly("curr_poly_string_list", &aGBZ1DSolver::curr_poly_string_list);
    m.def("winding_loop_gen", &Atlas::winding_loop_gen, "generate winding loop");
    // m.def("winding_number", &Atlas::winding_number);
}