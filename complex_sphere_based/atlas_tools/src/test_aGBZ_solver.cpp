/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-27
 * @lastModified  2023-07-27
 * Copyright © Department of Physics, Tsinghua University.  All rights reserved 
 */

#include "aGBZ_solvers.hpp"
#include <eigen3/Eigen/Core>
#include <math.h>

class TestAGBZSolver: public aGBZSolver<Atlas::ComplexType, 1, 1>
{
public:
    /* types */
    typedef aGBZSolver<Atlas::ComplexType, 1, 1>::ParamMatrixType ParamMatrixType;
    typedef aGBZSolver<Atlas::ComplexType, 1, 1>::DataPointViewType DataPointViewType;
    typedef aGBZSolver<Atlas::ComplexType, 1, 1>::PointMatrixType PointMatrixType;

    void sol_to_segments(){};
    void equation_update(){};
    bool solution_filter(DataPointViewType::PointDataType curr_sol){
        return true;
    }
    void full_solver_update_string(){};
    PointMatrixType continuation_solver_fun(const PointMatrixType & x){
        return PointMatrixType::Zero(3,2);
    };
};

int test_circle_loops()
{
    const size_t n_points = 200;
    TestAGBZSolver my_solver;
    Eigen::MatrixXcd param_list(n_points, 1);
    Eigen::VectorXcd beta_norm = {};
    param_list.col(0) = Eigen::VectorXcd::LinSpaced(n_points, 0, 2*M_PI);

    my_solver.set_param(param_list);

    do
    {
        if(my_solver.full_solver_flag)
        {

        }
        else
        {

        }
    }while(my_solver.next());
}
