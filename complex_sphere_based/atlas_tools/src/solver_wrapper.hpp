/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-08-20
 * @lastModified  2023-08-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#ifndef SOLVER_WRAPPER_HPP
#define SOLVER_WRAPPER_HPP
#include <pybind11/embed.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <eigen3/Eigen/Core>
#include <string>
#include <vector>

namespace py=pybind11;
using namespace py::literals;

py::list string_cast(std::vector<std::string> cpp_string_vec)
{
    py::list py_str_list;
    for(auto curr_str:cpp_string_vec)
    {
        py_str_list.append(py::cast(curr_str));
    }
}

void phcpy_solve(std::vector<std::string> poly_eqs, std::vector<std::string> var_names,
        Eigen::MatrixXcd & result_list)
{
    py::module_ mod_phcpy_solver = py::module_::import("phcpy.solver");
    py::list all_sols = mod_phcpy_solver.attr("solve")(
        string_cast(poly_eqs), "verbose"_a=false, "dictionary_output"_a=true
    );

    result_list.resize(all_sols.size(), var_names.size());

    // access each solutions
    for(size_t sol_id = 0; sol_id < all_sols.size(); sol_id++)
    {
        for(size_t var_id = 0; var_id < var_names.size(); var_id++)
        {
            py::dict curr_sol = all_sols[sol_id];
            result_list(sol_id, var_id) = curr_sol[py::cast(var_names[var_id])].cast<std::complex<double>>();
        }
    }
}

#endif