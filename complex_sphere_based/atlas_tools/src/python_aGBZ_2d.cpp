/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-08-16
 * @lastModified  2023-10-01
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "aGBZ_2d.hpp"
#include "poly_tools/poly_ring.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <functional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py=pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<aGBZ2DSolver::DataPointViewType::ScalarType>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<Atlas::IndexType>);
PYBIND11_MAKE_OPAQUE(std::vector<Atlas::ChartIdType>);
PYBIND11_MAKE_OPAQUE(std::vector<py::array_t<Atlas::ComplexType>>);

// double python_winding_phase(Atlas::RealType start_phase, Atlas::RealType end_phase, Atlas::RealType radius, Atlas::IndexType N_points, 
//         Atlas::RealType dphase_th, const std::function<void(const Atlas::ComplexVec&, Atlas::ComplexVec&)>& fun)
// double python_winding_phase(Atlas::RealType start_phase, Atlas::RealType end_phase, Atlas::RealType radius, Atlas::IndexType N_points, 
//         Atlas::RealType dphase_th, const py::function& fun)
// {
//     auto cpp_fun = [fun](const Atlas::ComplexVec& beta_loop, Atlas::ComplexVec& vals)
//     {
//         fun(beta_loop, vals);
//     };
//     return Atlas::winding_phase(start_phase, end_phase, radius, N_points, dphase_th, cpp_fun);
// }

// double python_winding_phase_with_param(Atlas::RealType start_phase, Atlas::RealType end_phase, Atlas::RealType radius, Atlas::IndexType N_points, 
//         Atlas::RealType dphase_th, py::array_t<Atlas::ComplexType> start_param,
//         const std::function<void(const Atlas::ComplexVec&, Atlas::ComplexVec&, typename std::vector<py::array_t<Atlas::ComplexType>> &,
//          py::array_t<Atlas::ComplexType>)> & fun)
// // double python_winding_phase_with_param(Atlas::RealType start_phase, Atlas::RealType end_phase, Atlas::RealType radius, Atlas::IndexType N_points, 
// //         Atlas::RealType dphase_th, py::array_t<Atlas::ComplexType> start_param,
// //         const py::function & fun)
// {
//     return Atlas::winding_phase<py::array_t<Atlas::ComplexType>>(start_phase, end_phase, radius, 
//         N_points, dphase_th, start_param, fun);
// }

PYBIND11_MAKE_OPAQUE(std::deque<Atlas::ZeroPointSegment>);

PYBIND11_MODULE(c_aGBZ_2D_solver, m){
    py::bind_vector<std::vector<aGBZ2DSolver::DataPointViewType::ScalarType>>(m, "aGBZ_2D_solver_ScalarVec");
    py::bind_vector<std::vector<int>>(m, "aGBZ_2D_solver_IntVec");
    py::bind_vector<std::vector<std::string>>(m, "aGBZ_2D_solver_StrVec");
    py::bind_vector<std::vector<Atlas::ChartIdType>>(m, "aGBZ_2D_solver_ChartIdVec");
    py::bind_vector<std::vector<Atlas::IndexType>>(m, "aGBZ_2D_solver_IndexVec");
    py::bind_vector<std::vector<py::array_t<Atlas::ComplexType>>>(m, "aGBZ_2D_solver_ComplexArrVec");

    // m.def("python_winding_phase", &python_winding_phase);
    // m.def("python_winding_phase_with_param", &python_winding_phase_with_param);

    py::class_<PolyTools::PolyDeque>(m, "aGBZ_2D_solver_PolyDeque")
        .def(py::init<>())
        .def("push_front", &PolyTools::PolyDeque::push_front)
        .def("get_poly", &PolyTools::PolyDeque::get_poly)
        .def("size", &PolyTools::PolyDeque::size)
        .def("push_back", &PolyTools::PolyDeque::push_back);
    
    m.def("aGBZ_2D_buchberger", &PolyTools::buchberger);

    py::class_<PolyTools::Laurant>(m, "aGBZ_2D_solver_Laurant")
        .def(py::init<PolyTools::IndexType>())
        .def_readonly("dim", &PolyTools::Laurant::dim)
        .def_readonly("num_max_orders", &PolyTools::Laurant::num_max_orders)
        .def_readonly("denom_orders", &PolyTools::Laurant::denom_orders)
        .def_readwrite("num", &PolyTools::Laurant::num)
        .def("reinit", &PolyTools::Laurant::reinit)
        .def("reduction", &PolyTools::Laurant::reduction)
        .def("set_Laurant", &PolyTools::Laurant::set_Laurant)
        .def("set_Laurant_by_terms", &PolyTools::Laurant::set_Laurant_by_terms)
        .def("flip_variable", &PolyTools::Laurant::flip_variable)
        .def("eval", &PolyTools::Laurant::eval);

    py::class_<aGBZ2DSolver>(m, "aGBZ2DSolver")
        .def(py::init<Atlas::IndexType, Atlas::RealType, Atlas::IndexType, 
            Atlas::IndexType, Atlas::RealType, Atlas::RealType>())
        .def("set_param", &aGBZ2DSolver::set_param)
        .def("set_char_poly_by_coeffs", &aGBZ2DSolver::set_char_poly_by_coeffs)
        .def("next", &aGBZ2DSolver::next)
        .def("continuation_solver_fun", &aGBZ2DSolver::continuation_solver_fun)
        .def_readonly("curr_initial_values", &aGBZ2DSolver::curr_initial_values)
        .def("insert_continuation_solver_results", &aGBZ2DSolver::insert_continuation_solver_results)
        .def("insert_full_solver_results", &aGBZ2DSolver::insert_full_solver_results)
        .def("suspend_solver", &aGBZ2DSolver::suspend_solver)
        .def("post_process", &aGBZ2DSolver::post_process)
        .def("loop_readout", &aGBZ2DSolver::loop_readout)
        .def("N_loops", [](aGBZ2DSolver & solver){return solver.all_loops.size();})
        .def("N_segments", [](aGBZ2DSolver & solver){return solver.all_segments.size();})
        .def("segment_readout", &aGBZ2DSolver::segment_readout)
        .def("raw_sol_readout", &aGBZ2DSolver::raw_sol_readout)
        .def("N_points", [](aGBZ2DSolver & solver){return solver.all_points.size();})
        // .def("sol_to_segments", &aGBZ2DSolver::sol_to_segments)
        // .def("segments_to_loops", &aGBZ2DSolver::segments_to_loops)
        .def("sol_to_segments_init", &aGBZ2DSolver::sol_to_segments_init)
        .def("GBZ_check_main_loop", &aGBZ2DSolver::GBZ_check_main_loop)
        .def("get_curr_param", &aGBZ2DSolver::get_curr_param)
        .def("set_char_poly", &aGBZ2DSolver::set_char_poly)
        .def_readonly("major_dim", &aGBZ2DSolver::major_dim)
        .def_readonly("solver_suspended", &aGBZ2DSolver::solver_suspended)
        .def_readonly("full_solver_flag", &aGBZ2DSolver::full_solver_flag)
        .def_readonly("param_ptr", &aGBZ2DSolver::param_ptr)
        .def_readonly("GBZ_check_curr_point", &aGBZ2DSolver::GBZ_check_curr_point)
        .def_readonly("GBZ_check_curr_ptr", &aGBZ2DSolver::GBZ_check_curr_ptr)
        .def_readonly("curr_poly_string_list", &aGBZ2DSolver::curr_poly_string_list);
    m.def("winding_loop_gen", &Atlas::winding_loop_gen, "generate winding loop");
    // m.def("winding_number", &Atlas::winding_number);

    py::class_<PolyTools::PolyLinkedList>(m, "aGBZ_2D_solver_PolyLinkedList")
        .def(py::init<PolyTools::IndexType>())
        // .def("reinit", &PolyTools::PolyLinkedList::reinit)
        .def("reinit", py::overload_cast<PolyTools::IndexType>(&PolyTools::PolyLinkedList::reinit))
        .def("remove_zeros", &PolyTools::PolyLinkedList::remove_zeros)
        .def("copy_to", &PolyTools::PolyLinkedList::copy_to)
        .def("to_str", &PolyTools::PolyLinkedList::to_str)
        .def("print_info", &PolyTools::PolyLinkedList::print_info)
        .def("destructive_add_self", &PolyTools::PolyLinkedList::destructive_add_self)
        .def("destructive_add", &PolyTools::PolyLinkedList::destructive_add)
        .def("add_self", &PolyTools::PolyLinkedList::add_self)
        .def("add", &PolyTools::PolyLinkedList::add)
        .def("scalar_mul_self", &PolyTools::PolyLinkedList::scalar_mul_self)
        .def("scalar_mul", &PolyTools::PolyLinkedList::scalar_mul)
        .def("scale_var_self", &PolyTools::PolyLinkedList::scale_var_self)
        .def("scale_var", &PolyTools::PolyLinkedList::scale_var)
        .def("neg_self", &PolyTools::PolyLinkedList::neg_self)
        .def("neg", &PolyTools::PolyLinkedList::neg)
        .def("destructive_subs_self", &PolyTools::PolyLinkedList::destructive_subs_self)
        .def("destructive_subs", &PolyTools::PolyLinkedList::destructive_subs)
        .def("subs_self", &PolyTools::PolyLinkedList::subs_self)
        .def("subs", &PolyTools::PolyLinkedList::subs)
        .def("derivative", &PolyTools::PolyLinkedList::derivative)
        .def("eval", &PolyTools::PolyLinkedList::eval)
        .def("eval_diff", &PolyTools::PolyLinkedList::eval_diff)
        .def("partial_eval", &PolyTools::PolyLinkedList::partial_eval)
        .def("batch_eval", &PolyTools::PolyLinkedList::batch_eval)
        .def("batch_add_elements", &PolyTools::PolyLinkedList::batch_add_elements)
        .def("batch_get_data", &PolyTools::PolyLinkedList::batch_get_data)
        .def("init_with_data", &PolyTools::PolyLinkedList::init_with_data)
        .def_readonly("dim", &PolyTools::PolyLinkedList::dim)
        .def_readonly("n_terms", &PolyTools::PolyLinkedList::n_terms);

    py::class_<GBZCondition1DChecker>(m, "GBZCondition1DChecker")
        .def(py::init<PolyTools::Laurant&, Atlas::IndexType>())
        .def("GBZ_condition_check", &GBZCondition1DChecker::GBZ_condition_check);

    py::class_<std::deque<Atlas::ZeroPointSegment>>(m, "AtlasZeroPointSegmentDeque")
        .def(py::init<>())
        .def("pop_front", &std::deque<Atlas::ZeroPointSegment>::pop_front)
        .def("pop_back", &std::deque<Atlas::ZeroPointSegment>::pop_back)
        .def("__len__", &std::deque<Atlas::ZeroPointSegment>::size)
        .def("__getitem__", [](const std::deque<Atlas::ZeroPointSegment>& self, size_t key){
            return self[key];
        })
        .def("__setitem__", [](std::deque<Atlas::ZeroPointSegment> & self, size_t key, const Atlas::ZeroPointSegment & new_point){
            self[key] = new_point;
        });

    py::class_<Atlas::ZeroPointSegment>(m, "AtlasZeroPointSegment")
        .def(py::init<Atlas::RealType>())
        .def_readonly("N_points", &Atlas::ZeroPointSegment::N_points)
        .def_readonly("left_point", &Atlas::ZeroPointSegment::left_point)
        .def_readonly("right_point", &Atlas::ZeroPointSegment::right_point)
        .def_readonly("left_index", &Atlas::ZeroPointSegment::left_index)
        .def_readonly("right_index", &Atlas::ZeroPointSegment::right_index);
    
    m.def("atlas_bubble_gen", &Atlas::bubble_gen);
    m.def("atlas_long_bubble_gen", &Atlas::long_bubble_gen);
    m.def("atlas_collect_zero_point", &Atlas::collect_zero_point);

}