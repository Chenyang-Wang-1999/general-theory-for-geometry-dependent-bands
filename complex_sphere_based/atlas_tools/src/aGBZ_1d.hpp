/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-20
 * @lastModified  2023-08-17
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#ifndef AGBZ_1D_HPP
#include "atlas/atlas.hpp"
#include "atlas/GBZ_manifolds.hpp"
#include "aGBZ_solvers.hpp"
#include "poly_tools/polys.hpp"
#include <Eigen/Core>
#include <algorithm>

class aGBZ1DSolver: public aGBZSolver<Atlas::ComplexType, 1, 1, 1>
{
public:
    /* configuration */
    Atlas::IndexType N_loop_points, N_detour;
    Atlas::RealType rel_r_detour;

    /* constructor */
    aGBZ1DSolver(Atlas::RealType critical_distance, Atlas::IndexType N_loop_points, 
            Atlas::IndexType N_detour, Atlas::RealType rel_r_detour): N_loop_points(N_loop_points), N_detour(N_detour), 
            rel_r_detour(rel_r_detour){
        this->critical_distance = critical_distance;
    }

    /* GBZ types*/
    typedef Atlas::GBZBaseManifold<1> DataPointViewType;
    typedef Atlas::GBZBaseManifold<1> PolyPointViewType;
    typedef Eigen::Matrix<DataPointViewType::ScalarType, Eigen::Dynamic, DataPointViewType::dim> PointMatrixType;

    /* characteristic polynomials */
    CharPolyFun<1> char_poly_fun;
    void set_char_poly(PolyTools::Laurant & char_poly_0)
    {
        char_poly_fun.set_char_poly(char_poly_0);
    }

    void set_char_poly_by_coeffs(const PolyTools::ScalarVec & coeffs, const std::vector<int> & orders)
    {
        PolyTools::Laurant char_poly(PolyPointViewType::dim);
        char_poly.set_Laurant_by_terms(coeffs, orders);
        set_char_poly(char_poly);
    }

    void equation_update()
    {
        char_poly_fun.curr_chart_labels_binary = solver_curr_chart_binary;
    }


    /* Full Solvers */
    void full_solver_update_string()
    {
        Atlas::ComplexType expiphi = (solver_curr_chart_binary == 0)? param_list(param_ptr,0): 1.0/param_list(param_ptr,0);
        if(char_poly_fun.curr_chart_labels_binary != solver_curr_chart_binary)
        {
            STDERR << "full_solver_update_string: Warning: full_solver_chart_ptr != char_poly_fun.curr_chart_labels_binary\n";
            assert(0);
        }
        PolyTools::PolyLinkedList new_poly(2);
        curr_poly_string_list[0] = char_poly_fun.get_current_laurant().num.to_str({"x", "y"});
        char_poly_fun.get_current_laurant().num.scale_var({1.0, expiphi}, new_poly);
        curr_poly_string_list[1] = new_poly.to_str({"x", "y"});
    }

    PointMatrixType continuation_solver_fun(const PointMatrixType & x)
    {
        /*
            f(x, y) = 0
            f(x, expiphi * y) = 0
            <=>
            g(x, 1/y) = f(x, y) = 0
            g(x, 1/expiphi * 1/y) = f(x, y) = 0
        */
        PointMatrixType result_mat(DataPointViewType::dim + 1, DataPointViewType::dim);

        PolyTools::VarScalarVec x_scalarvec(2);
        for(Atlas::IndexType j = 0; j<2; j++)
        {
            x_scalarvec[j] = x(0,j);
        }

        PolyTools::VarScalarVec x_another(x_scalarvec.begin(), x_scalarvec.end());

        Atlas::ComplexType expiphi = param_list(param_ptr, 0);
        Atlas::ComplexType scale_factor = expiphi;
        Atlas::IndexType poly_id = solver_curr_chart_binary;

        if(poly_id == 0)
        {
            scale_factor = expiphi;
        }
        else
        {
            scale_factor = 1.0/expiphi;
        }
        x_another[1] *= scale_factor;

        // Calculate values
        result_mat(0, 0) = char_poly_fun.get_current_laurant().num.eval(x_scalarvec);
        result_mat(0, 1) = char_poly_fun.get_current_laurant().num.eval(x_another);

        // Calculate jacobian
        result_mat(1, 0) = char_poly_fun.get_current_laurant().num.eval_diff({1,0}, x_scalarvec);
        result_mat(1, 1) = char_poly_fun.get_current_laurant().num.eval_diff({0,1}, x_scalarvec);
        result_mat(2, 0) = char_poly_fun.get_current_laurant().num.eval_diff({1,0}, x_another);
        result_mat(2, 1) = char_poly_fun.get_current_laurant().num.eval_diff({0,1}, x_another) * scale_factor;
        
        return result_mat;
    }

    /* solution check */
    bool solution_filter(DataPointViewType::PointDataType point_to_check)
    {
        return true;
    }

    bool GBZ_condition(DataPointViewType::PointDataType point_to_check, Atlas::IndexType curr_ptr)
    {
        /* creat viewer */
        DataPointViewType viewer;

        /* get corresponding laurant polynomial of the point */
        char_poly_fun.set_chart_labels(point_to_check.chart_labels);
        Atlas::ComplexType curr_expiphi = (point_to_check.chart_labels[0] == 0)? param_list[curr_ptr] : 1.0/param_list[curr_ptr];

        /* get evaluation loop */
        std::vector<Atlas::ComplexType> path_12, circ_2, inner_circ_2, path_21, circ_1, inner_circ_1;
        std::vector<Atlas::ComplexType> val_path_12, val_circ_2, val_inner_circ_2, val_path_21, val_circ_1, val_inner_circ_1;
        Atlas::winding_loop_gen(point_to_check.coords[1], curr_expiphi, rel_r_detour, N_loop_points, N_detour,
            path_12, circ_2, inner_circ_2, path_21, circ_1, inner_circ_1);

        /* evaluate */
        val_path_12.clear(); val_path_12.resize(path_12.size());
        val_path_21.clear(); val_path_21.resize(path_21.size());
        val_circ_1.clear(); val_circ_1.resize(circ_1.size());
        val_circ_2.clear(); val_circ_2.resize(circ_2.size());
        val_inner_circ_1.clear(); val_inner_circ_1.resize(inner_circ_1.size());
        val_inner_circ_2.clear(); val_inner_circ_2.resize(inner_circ_2.size());
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, path_12, val_path_12);
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, path_21, val_path_21);
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, circ_1, val_circ_1);
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, circ_2, val_circ_2);
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, inner_circ_1, val_inner_circ_1);
        eval_char_poly_around_loop<1>(char_poly_fun, 1, point_to_check, inner_circ_2, val_inner_circ_2);

        std::vector<std::vector<Atlas::ComplexType>*> inner_loop_vals = {&val_path_12, & val_inner_circ_2, & val_path_21, &val_inner_circ_1};
        std::vector<std::vector<Atlas::ComplexType>*> outer_loop_vals = {&val_path_12, & val_circ_2, & val_path_21, &val_circ_1};


        // 3. calculate winding numbers
        int outer_winding = Atlas::winding_number(outer_loop_vals);
        int inner_winding = Atlas::winding_number(inner_loop_vals);
        /* check GBZ condition */
        return ((outer_winding > 0 ) && (inner_winding < 0));
    }

    DataPointViewType::PointDataType point_another(DataPointViewType::PointDataType curr_point, Atlas::IndexType curr_ptr)
    {
        DataPointViewType viewer(curr_point.coords.data(), curr_point.chart_labels.data());
        DataPointViewType::PointDataType new_point;
        viewer.copy_data(new_point);

        // transform
        if(new_point.chart_labels[0] == 0)
        {
            new_point.coords[1] *= param_list[curr_ptr];
        }
        else
        {
            new_point.coords[1] /= param_list[curr_ptr];
        }
        return new_point;
    }

    void sol_to_segments()
    {
        // initialize the segment vector
        all_segments.clear();
        Atlas::IndexType segment_size = all_points[0].size();

        for(Atlas::IndexType segment_id = 0; segment_id < segment_size; segment_id++)
        {
            std::vector<DataPointViewType::PointDataType> segment1, segment2;
            segment1.clear(); segment2.clear();
            for(Atlas::IndexType curr_ptr = 0; curr_ptr < all_points.size(); curr_ptr ++)
            {
                if(GBZ_condition(all_points[curr_ptr][segment_id], curr_ptr))
                {
                    segment1.push_back(all_points[curr_ptr][segment_id]);
                    segment2.push_back(point_another(all_points[curr_ptr][segment_id], curr_ptr));
                }
                else
                {
                    if(segment1.size())
                    {
                        std::reverse(segment1.begin(), segment1.end());
                        all_segments.push_back(segment1);
                        all_segments.push_back(segment2);
                        segment1.clear();
                        segment2.clear();
                    }
                }
            }
            if(segment1.size())
            {
                std::reverse(segment1.begin(), segment1.end());
                all_segments.push_back(segment1);
                all_segments.push_back(segment2);
            }
        }
    }
};

#endif
