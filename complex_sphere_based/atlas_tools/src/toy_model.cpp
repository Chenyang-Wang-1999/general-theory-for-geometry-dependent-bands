/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-20
 * @lastModified  2023-07-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include <sstream>
#include <iostream>
#include "atlas/atlas.hpp"
#include "atlas/GBZ_manifolds.hpp"
#include "aGBZ_solvers.hpp"
#include "poly_tools/polys.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>

class ToyModelTest: public aGBZSolver<Atlas::ComplexType, 1, 1>
{
public:
    /* GBZ types*/
    typedef Atlas::GBZBaseManifold<1> DataPointViewType;
    typedef Eigen::Matrix<DataPointViewType::ScalarType, Eigen::Dynamic, DataPointViewType::dim> PointMatrixType;
    const std::vector<DataPointViewType::ChartLabelType> all_chart_labels = {{0}, {1}};

    
    void full_solver_update_string()
    {
        if(all_chart_labels[full_solver_chart_ptr][0] == 0)
        {
            curr_poly_string_list[0] = "x0 + x1;";
            std::ostringstream poly_string_stream;
            poly_string_stream << "x0 + x1 * " << param_list(param_ptr, 0) << ';';
            curr_poly_string_list[1] = poly_string_stream.str();
        }
        else
        {
            curr_poly_string_list[0] = "x0 * x1 + 1;";
            std::ostringstream poly_string_stream;
            poly_string_stream << "x0 * x1 *" << param_list(param_ptr, 0) << "+ 1" << ';';
            curr_poly_string_list[1] = poly_string_stream.str();
        }
    }

    PointMatrixType continuation_solver_fun(const PointMatrixType & x)
    {
        PointMatrixType result_mat(DataPointViewType::dim + 1, DataPointViewType::dim);
        result_mat.setZero();
        return result_mat;
    }
};

// int main()
// {
//     std::cout << ToyModelTest::DataPointViewType::dim << '\n';
//     return 0;
// }

namespace py=pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<ToyModelTest::DataPointViewType::ScalarType>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MODULE(test_toy_model, m){
    py::bind_vector<std::vector<ToyModelTest::DataPointViewType::ScalarType>>(m, "ScalarVec");
    py::bind_vector<std::vector<std::string>>(m, "StrVec");
    // py::class_<Atlas::GBZBaseManifold<1>>(m, "GBZBaseManifold1D")
    //     .def(py::init<>())
    //     .def("get_dim", ([](){return Atlas::GBZBaseManifold<1>::dim;}));
    py::class_<ToyModelTest>(m, "ToyModelTest")
        .def(py::init<>());
        // .def("set_param", &ToyModelTest::set_param)
        // .def_readonly("full_solver_flag", &ToyModelTest::full_solver_flag)
        // .def_readonly("curr_poly_string_list", &ToyModelTest::curr_poly_string_list);
}