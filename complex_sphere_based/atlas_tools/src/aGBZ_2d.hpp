/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-08-16
 * @lastModified  2023-08-16
 * Copyright © Department of Physics, Tsinghua University. All rights reserved
 */

#ifndef AGBZ_2D_HPP
#define AGBZ_2D_HPP

#include "aGBZ_solvers.hpp"
#include "poly_tools/polys.hpp"
#include <functional>

class aGBZ2DSolver : public aGBZSolver<Atlas::ComplexType, 2, 3, 2>
{
public:
    /* data structure:
       The four coordinates stand for E, beta_x, beta_y, beta_another 
       When "major_dim == 0", beta_another = betax_another, vice versa.
    */

    /* configuration */
    Atlas::IndexType major_dim;
    Atlas::IndexType N_loop_points;
    Atlas::IndexType N_detour_points;
    Atlas::RealType detour_phase_change;
    Atlas::RealType zero_val_th;

    /* constructor */
    aGBZ2DSolver(Atlas::IndexType major_dim, Atlas::RealType critical_distance, Atlas::IndexType N_loop_points, 
            Atlas::IndexType N_detour_points, Atlas::RealType detour_phase_change, Atlas::RealType zero_val_th): 
            major_dim(major_dim), N_loop_points(N_loop_points), N_detour_points(N_detour_points),
            detour_phase_change(detour_phase_change), zero_val_th(zero_val_th){
        this->critical_distance = critical_distance;
    }

    /* GBZ types*/
    typedef Atlas::GBZBaseManifold<3> DataPointViewType;
    typedef Atlas::GBZBaseManifold<2> PolyPointViewType;
    typedef Eigen::Matrix<DataPointViewType::ScalarType, Eigen::Dynamic, DataPointViewType::dim> PointMatrixType;
    typedef Eigen::Matrix<DataPointViewType::ScalarType, DataPointViewType::dim, 1> PointVecType;

    /* characteristic polynomials */
    CharPolyFun<2> char_poly_fun;
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
    {}


    /* Full Solvers */
    void full_solver_update_string()
    {
        if(major_dim == 0)
        {
            full_solver_update_string_x_major();
        }
        else if(major_dim == 1)
        {
            full_solver_update_string_y_major();
        }
        else
        {
            throw std::runtime_error("aGBZ2DSolver: major_dim must be 0 or 1");
        }
    }

    void full_solver_update_string_x_major()
    {
        /*
            f(x, y1, y2) = 0
            f(x, y1, y2*expiphi2) = 0
            f(x, y1*expiphi1, y3) = 0
            f(x, y1*expiphi1, y3*expiphi2) = 0
        */

        // 1. get current chart_labels
        DataPointViewType::ChartLabelType curr_chart_label;
        binary_to_chart_labels<3>(curr_chart_label, solver_curr_chart_binary);
        Atlas::IndexType poly1_chart, poly2_chart;
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[1]}, poly1_chart);
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[2]}, poly2_chart);

        Atlas::ComplexType scale_factor1 = (curr_chart_label[0])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);
        Atlas::ComplexType scale_factor2 = (curr_chart_label[1])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);
        Atlas::ComplexType scale_factor3 = (curr_chart_label[2])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);

        PolyTools::PolyLinkedList new_poly(3);

        // 2. equation 1
        char_poly_fun.curr_chart_labels_binary = poly1_chart;
        curr_poly_string_list[0] = char_poly_fun.get_current_laurant().num.to_str({"x", "y", "z"});
        char_poly_fun.get_current_laurant().num.scale_var({1.0, 1.0, scale_factor2}, new_poly);
        curr_poly_string_list[1] = new_poly.to_str({"x", "y", "z"});

        // 3. equation 2
        char_poly_fun.curr_chart_labels_binary = poly2_chart;
        char_poly_fun.get_current_laurant().num.scale_var({1.0, scale_factor1, 1.0}, new_poly);
        curr_poly_string_list[2] = new_poly.to_str({"x", "y", "w"});

        char_poly_fun.get_current_laurant().num.scale_var({1.0, scale_factor1, scale_factor3}, new_poly);
        curr_poly_string_list[3] = new_poly.to_str({"x", "y", "w"});

    }

    void full_solver_update_string_y_major()
    {
        /*
            f(x, y1, y2) = 0
            f(x, y1*expiphi1, y2) = 0
            f(x, y3, y2*expiphi2) = 0
            f(x, y3*expiphi1, y2*expiphi2) = 0
        */

        // 1. get current chart_labels
        DataPointViewType::ChartLabelType curr_chart_label;
        binary_to_chart_labels<3>(curr_chart_label, solver_curr_chart_binary);
        Atlas::IndexType poly1_chart, poly2_chart;
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[1]}, poly1_chart);
        chart_labels_to_binary<2>({curr_chart_label[2], curr_chart_label[1]}, poly2_chart);

        Atlas::ComplexType scale_factor1 = (curr_chart_label[0])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);
        Atlas::ComplexType scale_factor2 = (curr_chart_label[1])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);
        Atlas::ComplexType scale_factor3 = (curr_chart_label[2])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);

        PolyTools::PolyLinkedList new_poly(3);

        // 2. equation 1
        char_poly_fun.curr_chart_labels_binary = poly1_chart;
        curr_poly_string_list[0] = char_poly_fun.get_current_laurant().num.to_str({"x", "y", "z"});
        char_poly_fun.get_current_laurant().num.scale_var({1.0, scale_factor1, 1.0}, new_poly);
        curr_poly_string_list[1] = new_poly.to_str({"x", "y", "z"});

        // 3. equation 2
        char_poly_fun.curr_chart_labels_binary = poly2_chart;
        char_poly_fun.get_current_laurant().num.scale_var({1.0, 1.0, scale_factor2}, new_poly);
        curr_poly_string_list[2] = new_poly.to_str({"x", "w", "z"});

        char_poly_fun.get_current_laurant().num.scale_var({1.0, scale_factor3, scale_factor2}, new_poly);
        curr_poly_string_list[3] = new_poly.to_str({"x", "w", "z"});

    }

    PointMatrixType continuation_solver_fun(const PointMatrixType & x)
    {
        if(major_dim == 0)
        {
            return continuation_solver_fun_x_major(x);
        }
        else if(major_dim == 1)
        {
            return continuation_solver_fun_y_major(x);
        }
        else
        {
            throw std::runtime_error("aGBZ2DSolver: major_dim must be 0 or 1");
        }
    }

    PointMatrixType continuation_solver_fun_x_major(const PointMatrixType & x)
    {
        /*
            f(x, y1, y2) = 0
            f(x, y1, y2*expiphi2) = 0
            f(x, y1*expiphi1, y3) = 0
            f(x, y1*expiphi1, y3*expiphi2) = 0
        */
        PointMatrixType result_mat(DataPointViewType::dim + 1, DataPointViewType::dim);

        // 1. get current chart_labels
        DataPointViewType::ChartLabelType curr_chart_label;
        binary_to_chart_labels<3>(curr_chart_label, solver_curr_chart_binary);
        Atlas::IndexType poly1_chart, poly2_chart;
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[1]}, poly1_chart);
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[2]}, poly2_chart);

        Atlas::ComplexType scale_factor1 = (curr_chart_label[0])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);
        Atlas::ComplexType scale_factor2 = (curr_chart_label[1])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);
        Atlas::ComplexType scale_factor3 = (curr_chart_label[2])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);

        // 2. evaluate poly1
        char_poly_fun.curr_chart_labels_binary = poly1_chart;
        result_mat(0,0) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1), x(2)});
        result_mat(0,1) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1), x(2) * scale_factor2});
        result_mat(1,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1), x(2)});
        result_mat(1,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1), x(2)});
        result_mat(1,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1), x(2)});
        result_mat(1,3) = 0;

        result_mat(2,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1), x(2) * scale_factor2});
        result_mat(2,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1), x(2) * scale_factor2});
        result_mat(2,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1), x(2) * scale_factor2}) * scale_factor2;
        result_mat(2,3) = 0;

        // 3. evaluate poly2
        char_poly_fun.curr_chart_labels_binary = poly2_chart;
        result_mat(0,2) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1) * scale_factor1, x(3)});
        result_mat(0,3) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1) * scale_factor1, x(3) * scale_factor3});

        result_mat(3,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1) * scale_factor1, x(3)});
        result_mat(3,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1) * scale_factor1, x(3)}) * scale_factor1;
        result_mat(3,3) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1) * scale_factor1, x(3)});
        result_mat(3,2) = 0;

        result_mat(4,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1) * scale_factor1, x(3) * scale_factor3});
        result_mat(4,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1) * scale_factor1, x(3) * scale_factor3}) * scale_factor1;
        result_mat(4,3) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1) * scale_factor1, x(3) * scale_factor3}) * scale_factor3;
        result_mat(4,2) = 0;

        return result_mat;
    }

    PointMatrixType continuation_solver_fun_y_major(const PointMatrixType & x)
    {
        /*
            f(x, y1, y2) = 0
            f(x, y1*expiphi1, y2) = 0
            f(x, y3, y2*expiphi2) = 0
            f(x, y3*expiphi1, y2*expiphi2) = 0
        */
        PointMatrixType result_mat(DataPointViewType::dim + 1, DataPointViewType::dim);

        // 1. get current chart_labels
        DataPointViewType::ChartLabelType curr_chart_label;
        binary_to_chart_labels<3>(curr_chart_label, solver_curr_chart_binary);
        Atlas::IndexType poly1_chart, poly2_chart;
        chart_labels_to_binary<2>({curr_chart_label[0], curr_chart_label[1]}, poly1_chart);
        chart_labels_to_binary<2>({curr_chart_label[2], curr_chart_label[1]}, poly2_chart);

        Atlas::ComplexType scale_factor1 = (curr_chart_label[0])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);
        Atlas::ComplexType scale_factor2 = (curr_chart_label[1])? 1.0/param_list(param_ptr,1):param_list(param_ptr,1);
        Atlas::ComplexType scale_factor3 = (curr_chart_label[2])? 1.0/param_list(param_ptr,0):param_list(param_ptr,0);

        // 2. evaluate poly1
        char_poly_fun.curr_chart_labels_binary = poly1_chart;
        result_mat(0,0) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1), x(2)});
        result_mat(0,1) = char_poly_fun.get_current_laurant().num.eval({x(0), x(1)*scale_factor1, x(2)});
        result_mat(1,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1), x(2)});
        result_mat(1,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1), x(2)});
        result_mat(1,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1), x(2)});
        result_mat(1,3) = 0;

        result_mat(2,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(1)*scale_factor1, x(2)});
        result_mat(2,1) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(1)*scale_factor1, x(2)}) * scale_factor1;
        result_mat(2,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(1)*scale_factor1, x(2)});
        result_mat(2,3) = 0;

        // 3. evaluate poly2
        char_poly_fun.curr_chart_labels_binary = poly2_chart;
        result_mat(0,2) = char_poly_fun.get_current_laurant().num.eval({x(0), x(3), x(2) * scale_factor2});
        result_mat(0,3) = char_poly_fun.get_current_laurant().num.eval({x(0), x(3) * scale_factor3, x(2) * scale_factor2});

        result_mat(3,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(3), x(2) * scale_factor2});
        result_mat(3,3) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(3), x(2) * scale_factor2});
        result_mat(3,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(3), x(2) * scale_factor2}) * scale_factor2;
        result_mat(3,1) = 0;

        result_mat(4,0) = char_poly_fun.get_current_laurant().num.eval_diff({1, 0, 0}, {x(0), x(3) * scale_factor3, x(2) * scale_factor2});
        result_mat(4,3) = char_poly_fun.get_current_laurant().num.eval_diff({0, 1, 0}, {x(0), x(3) * scale_factor3, x(2) * scale_factor2}) * scale_factor3;
        result_mat(4,2) = char_poly_fun.get_current_laurant().num.eval_diff({0, 0, 1}, {x(0), x(3) * scale_factor3, x(2) * scale_factor2}) * scale_factor2;
        result_mat(4,1) = 0;

        return result_mat;
    }

    /* solution check */
    bool solution_filter(DataPointViewType::PointDataType point_to_check)
    {
        return true;
    }

    bool GBZ_condition(DataPointViewType::PointDataType point_to_check, Atlas::IndexType curr_ptr)
    {
        // STDERR << "coords:";
        // print_vector(point_to_check.coords);
        // STDERR << "\ncharts:";
        // print_vector(point_to_check.chart_labels);
        // STDERR << "\n";
        // STDERR << "curr_ptr:" << curr_ptr << '\n';
        if(major_dim == 0)
        {
            return GBZ_condition_x_major(point_to_check, curr_ptr);
        }
        else if(major_dim == 1)
        {
            return GBZ_condition_y_major(point_to_check, curr_ptr);
        }
        else
        {
            throw std::runtime_error("aGBZ2DSolver: major_dim must be 0 or 1");
        }
    }

    bool GBZ_condition_y_major(DataPointViewType::PointDataType point_to_check, Atlas::IndexType curr_ptr)
    {

        // winding loop
        int outer_winding, inner_winding;
        Atlas::ComplexType scale_factor = (point_to_check.chart_labels[1])? 1.0/param_list(curr_ptr, 1):param_list(curr_ptr, 1);

        /* 1. get the winding number of the first beta1 */

        // 1.1 set chart labels
        char_poly_fun.set_chart_labels({point_to_check.chart_labels[0], point_to_check.chart_labels[1]});


        // 1.3 evaluate on each segments
        PolyPointViewType::PointDataType curr_point;
        curr_point.chart_labels[0] = point_to_check.chart_labels[0];
        curr_point.chart_labels[1] = point_to_check.chart_labels[1];
        curr_point.coords[0] = point_to_check.coords[0];
        curr_point.coords[1] = point_to_check.coords[1];
        curr_point.coords[2] = point_to_check.coords[2];

        std::function<void(const Atlas::ComplexVec &, Atlas::ComplexVec &)> char_poly_eval = [this, &curr_point](
            const Atlas::ComplexVec & beta_list, Atlas::ComplexVec & vals
        ){
            eval_char_poly_around_loop<2>(this->char_poly_fun, 1, curr_point, beta_list, vals);
        };

        // 1.4 calculate winding number 
        bool success = Atlas::winding_phase_auto_detour(curr_point.coords[1], detour_phase_change, zero_val_th, 
                N_loop_points, N_detour_points, char_poly_eval, outer_winding, inner_winding);

        if(!success)
        {
            return false;
        }
        if(outer_winding > 0 && inner_winding < 0)
        {}
        else
        {
            return false;
        }


        /* 2. get the winding number of the second beta2 */
        // 2.1 set chart labels
        char_poly_fun.set_chart_labels({point_to_check.chart_labels[2], point_to_check.chart_labels[1]});

        // 2.3 evaluate on each segments
        curr_point.chart_labels[0] = point_to_check.chart_labels[2];
        curr_point.chart_labels[1] = point_to_check.chart_labels[1];
        curr_point.coords[0] = point_to_check.coords[0];
        curr_point.coords[1] = point_to_check.coords[3];
        curr_point.coords[2] = point_to_check.coords[2] * scale_factor;

        // 2.4 calculate winding number 
        success =  Atlas::winding_phase_auto_detour(curr_point.coords[1], detour_phase_change, zero_val_th,
                N_loop_points, N_detour_points, char_poly_eval, outer_winding, inner_winding);

        if(! success)
        {
            return false;
        }

        if(outer_winding > 0 && inner_winding < 0)
        {}
        else
        {
            return false;
        }
 
        /* 3. get the winding loop around beta1 */

        return true;
    }

    bool GBZ_condition_x_major(DataPointViewType::PointDataType point_to_check, Atlas::IndexType curr_ptr)
    {

        // winding loop
        int outer_winding, inner_winding;
        Atlas::ComplexType scale_factor = (point_to_check.chart_labels[0])? 1.0/param_list(curr_ptr, 0):param_list(curr_ptr, 0);

        /* 1. get the winding number of the first beta2 */
        // 1.1 set chart labels
        char_poly_fun.set_chart_labels({point_to_check.chart_labels[0], point_to_check.chart_labels[1]});


        // 1.3 evaluate on each segments
        PolyPointViewType::PointDataType curr_point;
        curr_point.chart_labels[0] = point_to_check.chart_labels[0];
        curr_point.chart_labels[1] = point_to_check.chart_labels[1];
        curr_point.coords[0] = point_to_check.coords[0];
        curr_point.coords[1] = point_to_check.coords[1];
        curr_point.coords[2] = point_to_check.coords[2];
        std::function<void(const Atlas::ComplexVec &, Atlas::ComplexVec &)> char_poly_eval = [this, &curr_point](
            const Atlas::ComplexVec & beta_list, Atlas::ComplexVec & vals
        ){
            eval_char_poly_around_loop<2>(this->char_poly_fun, 2, curr_point, beta_list, vals);
        };

        // 1.4 calculate winding number 
        bool success =  Atlas::winding_phase_auto_detour(curr_point.coords[2], detour_phase_change, zero_val_th, 
            N_loop_points, N_detour_points, char_poly_eval, outer_winding, inner_winding);
        if(! success)
        {
            return false;
        }
        
        if(outer_winding > 0 && inner_winding < 0)
        {}
        else
        {
            return false;
        }


        /* 2. get the winding number of the second beta2 */
        // 2.1 set chart labels
        char_poly_fun.set_chart_labels({point_to_check.chart_labels[0], point_to_check.chart_labels[2]});

        // 2.3 evaluate on each segments
        curr_point.chart_labels[0] = point_to_check.chart_labels[0];
        curr_point.chart_labels[1] = point_to_check.chart_labels[2];
        curr_point.coords[0] = point_to_check.coords[0];
        curr_point.coords[1] = point_to_check.coords[1] * scale_factor;
        curr_point.coords[2] = point_to_check.coords[3];

        // 2.4 calculate winding number 
        success =  Atlas::winding_phase_auto_detour(curr_point.coords[2], detour_phase_change, zero_val_th, 
            N_loop_points, N_detour_points, char_poly_eval, outer_winding, inner_winding);
        if(! success)
        {
            return false;
        }

        if(outer_winding > 0 && inner_winding < 0)
        {}
        else
        {
            return false;
        }
 
        /* 3. get the winding loop around beta1 */

        return true;
    }


    void get_other_points(const DataPointViewType::PointDataType &curr_point, Atlas::IndexType curr_ptr, 
        PolyPointViewType::PointDataType & poly_point1, 
        PolyPointViewType::PointDataType & poly_point2,
        PolyPointViewType::PointDataType & poly_point3,
        PolyPointViewType::PointDataType & poly_point4)
    {
        if(major_dim == 0)
        {
            get_other_points_x_major(curr_point, curr_ptr, poly_point1, poly_point2, poly_point3, poly_point4);
        }
        else if(major_dim == 1)
        {
            get_other_points_y_major(curr_point, curr_ptr, poly_point1, poly_point2, poly_point3, poly_point4);
        }
        else
        {
            throw std::runtime_error("aGBZ2DSolver: major_dim must be 0 or 1");
        }
    }

    void get_other_points_x_major(const DataPointViewType::PointDataType &curr_point, Atlas::IndexType curr_ptr, 
        PolyPointViewType::PointDataType & poly_point1, 
        PolyPointViewType::PointDataType & poly_point2,
        PolyPointViewType::PointDataType & poly_point3,
        PolyPointViewType::PointDataType & poly_point4)
    {
        Atlas::ComplexType expiphi1 = (curr_point.chart_labels[0]==0)? param_list(curr_ptr,0): 1.0/param_list(curr_ptr,0);
        Atlas::ComplexType expiphi2 = (curr_point.chart_labels[1]==0)? param_list(curr_ptr,1): 1.0/param_list(curr_ptr,1);
        Atlas::ComplexType expiphi3 = (curr_point.chart_labels[2]==0)? param_list(curr_ptr,1): 1.0/param_list(curr_ptr,1);

        poly_point1.chart_labels[0] = curr_point.chart_labels[0];
        poly_point1.chart_labels[1] = curr_point.chart_labels[1];
        poly_point1.coords[0] = curr_point.coords[0];
        poly_point1.coords[1] = curr_point.coords[1];
        poly_point1.coords[2] = curr_point.coords[2];

        poly_point2.chart_labels[0] = curr_point.chart_labels[0];
        poly_point2.chart_labels[1] = curr_point.chart_labels[1];
        poly_point2.coords[0] = curr_point.coords[0];
        poly_point2.coords[1] = curr_point.coords[1];
        poly_point2.coords[2] = curr_point.coords[2] * expiphi2;


        poly_point3.chart_labels[0] = curr_point.chart_labels[0];
        poly_point3.chart_labels[1] = curr_point.chart_labels[2];
        poly_point3.coords[0] = curr_point.coords[0];
        poly_point3.coords[1] = curr_point.coords[1] * expiphi1;
        poly_point3.coords[2] = curr_point.coords[3];


        poly_point4.chart_labels[0] = curr_point.chart_labels[0];
        poly_point4.chart_labels[1] = curr_point.chart_labels[2];
        poly_point4.coords[0] = curr_point.coords[0];
        poly_point4.coords[1] = curr_point.coords[1] * expiphi1;
        poly_point4.coords[2] = curr_point.coords[3] * expiphi3;

    }

    void get_other_points_y_major(const DataPointViewType::PointDataType &curr_point, Atlas::IndexType curr_ptr, 
        PolyPointViewType::PointDataType & poly_point1, 
        PolyPointViewType::PointDataType & poly_point2,
        PolyPointViewType::PointDataType & poly_point3,
        PolyPointViewType::PointDataType & poly_point4)
    {
        Atlas::ComplexType expiphi1 = (curr_point.chart_labels[0]==0)? param_list(curr_ptr,0): 1.0/param_list(curr_ptr,0);
        Atlas::ComplexType expiphi2 = (curr_point.chart_labels[1]==0)? param_list(curr_ptr,1): 1.0/param_list(curr_ptr,1);
        Atlas::ComplexType expiphi3 = (curr_point.chart_labels[2]==0)? param_list(curr_ptr,0): 1.0/param_list(curr_ptr,0);

        poly_point1.chart_labels[0] = curr_point.chart_labels[0];
        poly_point1.chart_labels[1] = curr_point.chart_labels[1];
        poly_point1.coords[0] = curr_point.coords[0];
        poly_point1.coords[1] = curr_point.coords[1];
        poly_point1.coords[2] = curr_point.coords[2];

        poly_point2.chart_labels[0] = curr_point.chart_labels[0];
        poly_point2.chart_labels[1] = curr_point.chart_labels[1];
        poly_point2.coords[0] = curr_point.coords[0];
        poly_point2.coords[1] = curr_point.coords[1] * expiphi1;
        poly_point2.coords[2] = curr_point.coords[2];


        poly_point3.chart_labels[0] = curr_point.chart_labels[2];
        poly_point3.chart_labels[1] = curr_point.chart_labels[1];
        poly_point3.coords[0] = curr_point.coords[0];
        poly_point3.coords[1] = curr_point.coords[3];
        poly_point3.coords[2] = curr_point.coords[2] * expiphi2;


        poly_point4.chart_labels[0] = curr_point.chart_labels[2];
        poly_point4.chart_labels[1] = curr_point.chart_labels[1];
        poly_point4.coords[0] = curr_point.coords[0];
        poly_point4.coords[1] = curr_point.coords[3] * expiphi3;
        poly_point4.coords[2] = curr_point.coords[2] * expiphi2;

    }

    /* tools for sol_to_segments */
    PointVecType GBZ_check_curr_point;
    std::vector<PolyPointViewType::PointDataType> segment1;
    std::vector<PolyPointViewType::PointDataType> segment2;
    std::vector<PolyPointViewType::PointDataType> segment3;
    std::vector<PolyPointViewType::PointDataType> segment4;

    Atlas::IndexType GBZ_check_segment_id;
    Atlas::IndexType GBZ_check_curr_ptr;

    bool sol_to_segments_init()
    {
        // initialize the segment vector
        // true: need to check aGBZ
        all_segments.clear();
        GBZ_check_segment_id = 0;
        GBZ_check_curr_ptr = 0;
        segment1.clear(); segment2.clear(); segment3.clear(); segment4.clear();

        while(true)
        {
            if(GBZ_condition(all_points[GBZ_check_curr_ptr][GBZ_check_segment_id], GBZ_check_curr_ptr))
            {
                DataPointViewType data_viewer(all_points[GBZ_check_curr_ptr][GBZ_check_segment_id].coords.data(),
                                            all_points[GBZ_check_curr_ptr][GBZ_check_segment_id].chart_labels.data());
                DataPointViewType::PointDataType new_data;
                data_viewer.copy_data(new_data);
                data_viewer.reinit_data(new_data.coords.data(), new_data.chart_labels.data());
                data_viewer.to_chart({0,0,0});
                GBZ_check_curr_point = Eigen::Map<PointVecType>(new_data.coords.data());
                return true;
            }
            else
            {
                GBZ_check_failed();
            }
            bool all_step_finished = move_next();
            if(all_step_finished)
            {
                return false;
            }
        }
    }


    void GBZ_check_ok()
    {
        PolyPointViewType::PointDataType new_point1, new_point2, new_point3, new_point4;
        get_other_points(all_points[GBZ_check_curr_ptr][GBZ_check_segment_id], GBZ_check_curr_ptr, 
            new_point1, new_point2, new_point3, new_point4);
        segment1.push_back(new_point1);
        segment2.push_back(new_point2);
        segment3.push_back(new_point3);
        segment4.push_back(new_point4);
    }

    void GBZ_check_failed()
    {
        if(segment1.size())
        {
            std::reverse(segment1.begin(), segment1.end());
            std::reverse(segment2.begin(), segment2.end());
            all_segments.push_back(segment1);
            all_segments.push_back(segment2);
            all_segments.push_back(segment3);
            all_segments.push_back(segment4);
            segment1.clear();
            segment2.clear();
            segment3.clear();
            segment4.clear();
        }
    }

    bool move_next()
    {
        GBZ_check_curr_ptr ++;
        if(GBZ_check_curr_ptr >= all_points.size())
        {
            if(segment1.size())
            {
                std::reverse(segment1.begin(), segment1.end());
                std::reverse(segment2.begin(), segment2.end());
                all_segments.push_back(segment1);
                all_segments.push_back(segment2);
                all_segments.push_back(segment3);
                all_segments.push_back(segment4);
            }

            GBZ_check_segment_id++;
            GBZ_check_curr_ptr = 0;

            segment1.clear(); segment2.clear(); segment3.clear(); segment4.clear();

        }

        if(GBZ_check_segment_id >= all_points[0].size())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool GBZ_check_main_loop(bool GBZ_check_result)
    {
        // check current result
        if(GBZ_check_result)
        {
            GBZ_check_ok();
        }
        else
        {
            GBZ_check_failed();
        }

        // continue loop        
        bool GBZ_checker_continue = !move_next();
        while(GBZ_checker_continue)
        {
            if(GBZ_condition(all_points[GBZ_check_curr_ptr][GBZ_check_segment_id], GBZ_check_curr_ptr))
            {
                DataPointViewType data_viewer(all_points[GBZ_check_curr_ptr][GBZ_check_segment_id].coords.data(),
                                            all_points[GBZ_check_curr_ptr][GBZ_check_segment_id].chart_labels.data());
                DataPointViewType::PointDataType new_data;
                data_viewer.copy_data(new_data);
                data_viewer.reinit_data(new_data.coords.data(), new_data.chart_labels.data());
                data_viewer.to_chart({0,0,0});
                GBZ_check_curr_point = Eigen::Map<PointVecType>(data_viewer.coord_ptr, data_viewer.dim);
                return true;
            }
            else
            {
                GBZ_check_failed();
            }
            bool all_step_finished = move_next();
            if(all_step_finished)
            {
                return false;
            }
        }
        return false;
    }

    void sol_to_segments()
    {
        // initialize the segment vector
        all_segments.clear();
        Atlas::IndexType segment_size = all_points[0].size();

        for(Atlas::IndexType segment_id = 0; segment_id < segment_size; segment_id++)
        {
            std::vector<PolyPointViewType::PointDataType> segment1, segment2, segment3, segment4;
            segment1.clear(); segment2.clear(); segment3.clear(); segment4.clear();
            for(Atlas::IndexType curr_ptr = 0; curr_ptr < all_points.size(); curr_ptr ++)
            {
                if(GBZ_condition(all_points[curr_ptr][segment_id], curr_ptr))
                {
                    PolyPointViewType::PointDataType new_point1, new_point2, new_point3, new_point4;
                    get_other_points(all_points[curr_ptr][segment_id], curr_ptr, new_point1, new_point2, new_point3, new_point4);
                    segment1.push_back(new_point1);
                    segment2.push_back(new_point2);
                    segment3.push_back(new_point3);
                    segment4.push_back(new_point4);
                }
                else
                {
                    if(segment1.size())
                    {
                        std::reverse(segment1.begin(), segment1.end());
                        std::reverse(segment2.begin(), segment2.end());
                        all_segments.push_back(segment1);
                        all_segments.push_back(segment2);
                        all_segments.push_back(segment3);
                        all_segments.push_back(segment4);
                        segment1.clear();
                        segment2.clear();
                        segment3.clear();
                        segment4.clear();
                    }
                }
            }
            if(segment1.size())
            {
                std::reverse(segment1.begin(), segment1.end());
                std::reverse(segment2.begin(), segment2.end());
                all_segments.push_back(segment1);
                all_segments.push_back(segment2);
                all_segments.push_back(segment3);
                all_segments.push_back(segment4);
            }
        }
    }
};

class GBZCondition1DChecker
{
public:
    GBZCondition1DChecker(PolyTools::Laurant & char_poly, Atlas::IndexType which_var):
        char_poly_ptr(&char_poly), which_var(which_var), curr_x(3){
            if(which_var == 1)
            {
                another_var = 2;
            }
            else if(which_var == 2)
            {
                another_var = 1;
            }
        }
    


    bool GBZ_condition_check(Atlas::ComplexType E_val, Atlas::ComplexType major_beta, Atlas::ComplexType beta, 
        Atlas::RealType detour_phase_change, Atlas::IndexType N_points, Atlas::IndexType N_detour, 
        Atlas::RealType zero_val_th)
    {
        curr_x[0] = E_val;
        curr_x[which_var] = major_beta;

        std::function<void(const Atlas::ComplexVec &, Atlas::ComplexVec &)> eval_fun = [this](
            const Atlas::ComplexVec & beta_loop, Atlas::ComplexVec & val
        ){this->eval_along_loop(beta_loop, val);};

        int outer_winding, inner_winding;
        bool success = Atlas::winding_phase_auto_detour(beta, detour_phase_change, zero_val_th, N_points, N_detour, 
            eval_fun, outer_winding, inner_winding);
        // STDOUT << outer_winding << ',' << inner_winding << '\n';
        if(! success)
        {
            return false;
        }


        if(outer_winding > 0 && inner_winding < 0)
        {
            return true;
        }
        else
        {
            return false;
        }

    }

private:
    PolyTools::Laurant * char_poly_ptr=NULL;
    Atlas::IndexType which_var;
    Atlas::IndexType another_var;
    Atlas::ComplexVec curr_x;
    void eval_along_loop(const Atlas::ComplexVec & beta_loop, Atlas::ComplexVec & vals)
    {
        for(size_t j = 0; j < beta_loop.size(); j++)
        {
            curr_x[another_var] = beta_loop[j];
            vals[j] = char_poly_ptr->eval(curr_x);
        }
    }
};

#endif // AGBZ_2D_HPP