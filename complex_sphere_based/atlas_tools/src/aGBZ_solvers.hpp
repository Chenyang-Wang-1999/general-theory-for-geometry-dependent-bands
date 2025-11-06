/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-19
 * @lastModified  2023-09-18
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#ifndef AGBZ_SOLVERS_HPP
#define AGBZ_SOLVERS_HPP

#include "debug_tools.hpp"
#include "atlas/atlas.hpp"
#include "atlas/GBZ_manifolds.hpp"
#include "poly_tools/polys.hpp"
#include <eigen3/Eigen/Core>
#include <string>
#include <list>

#define AGBZ_UNIQUE_TOL 1e-8

template<Atlas::IndexType _GBZ_dim>
void chart_labels_to_binary(const std::array<Atlas::ChartIdType, _GBZ_dim> & chart_labels,
            unsigned int & chart_labels_bin)
{
    chart_labels_bin = 0;
    for(unsigned int i = 0; i < _GBZ_dim; i++)
    {
        chart_labels_bin += (chart_labels[i] << i);
    }
}

template<Atlas::IndexType _GBZ_dim>
void binary_to_chart_labels(std::array<Atlas::ChartIdType, _GBZ_dim> & chart_labels, 
        unsigned int chart_labels_bin)
{
    for(unsigned int i = 0; i < _GBZ_dim; i++)
    {
        chart_labels[i] = chart_labels_bin % 2;
        chart_labels_bin >>= 1;
    }
}

template<Atlas::IndexType _GBZ_dim>
class CharPolyFun
{
public:
    std::array<PolyTools::Laurant*, 0x1<<_GBZ_dim> char_poly_list;
    unsigned int curr_chart_labels_binary = 0;
    CharPolyFun()
    {
        char_poly_list.fill(NULL);
    }

    CharPolyFun(PolyTools::Laurant & char_poly):CharPolyFun()
    {
        set_char_poly(char_poly);
    }

    ~CharPolyFun()
    {
        clear_char_poly();
    }

    void set_chart_labels(const typename Atlas::GBZBaseManifold<_GBZ_dim>::ChartLabelType & chart_labels)
    {
        chart_labels_to_binary<_GBZ_dim>(chart_labels, curr_chart_labels_binary);
    }

    void get_chart_labels(typename Atlas::GBZBaseManifold<_GBZ_dim>::ChartLabelType & chart_labels)
    {
        binary_to_chart_labels<_GBZ_dim>(chart_labels, curr_chart_labels_binary);
    }

    void clear_char_poly()
    {
        for(auto it = char_poly_list.begin(); it != char_poly_list.end(); it++)
        {
            if((*it) != NULL)
            {
                delete (*it);
                (*it) = NULL;
            }
        }

    }

    void update_polynomial(unsigned int chart_labels_bin)
    {
        /*
            find all ones and flip them one by one, until a known polynomial is reached
        */
       if(char_poly_list[chart_labels_bin] == NULL)
       {
            for(unsigned int curr_position = 0; curr_position < _GBZ_dim; curr_position ++)
            {
                if((chart_labels_bin >> curr_position)%2)
                {
                    unsigned int new_chart_labels = (chart_labels_bin - (0x1<<curr_position));
                    update_polynomial(new_chart_labels);
                    if(char_poly_list[chart_labels_bin] == NULL)
                    {
                        char_poly_list[chart_labels_bin] = new PolyTools::Laurant(_GBZ_dim + 1);
                        char_poly_list[new_chart_labels] -> flip_variable(curr_position + 1, *char_poly_list[chart_labels_bin]);
                    }
                }
            }
       }
    }

    void set_char_poly(PolyTools::Laurant & char_poly)
    {
        clear_char_poly();

        // 1. set initial value
        char_poly_list[0] = new PolyTools::Laurant(char_poly);

        // 2. update each elements
        for(unsigned int chart_labels_int = 0; chart_labels_int < (0x1<<_GBZ_dim); chart_labels_int ++)
        {
            update_polynomial(chart_labels_int);
        }
    }

    PolyTools::VarScalar eval(const typename Atlas::GBZBaseManifold<_GBZ_dim>::CoordType & coords)
    {
        PolyTools::VarScalarVec x_arr(coords.begin(), coords.end());
        return char_poly_list[curr_chart_labels_binary] -> eval(x_arr);
    }

    PolyTools::VarScalar eval(const typename Atlas::GBZBaseManifold<_GBZ_dim>::PointDataType & curr_point)
    {
        set_chart_labels(curr_point.chart_labels);
        return eval(curr_point.coords);
    }

    PolyTools::Laurant & get_current_laurant()
    {
        return *(char_poly_list[curr_chart_labels_binary]);
    }
};



template<typename ParamType, Atlas::IndexType param_dim, Atlas::IndexType _data_GBZ_dim, Atlas::IndexType _poly_GBZ_dim>
class aGBZSolver
{
public:

    /* configurations */
    Atlas::RealType critical_distance; // critical distance for continuation solver

    /* Parameters */
    typedef Eigen::Matrix<ParamType, Eigen::Dynamic, param_dim> ParamMatrixType;
    ParamMatrixType param_list;
    Atlas::IndexType param_ptr;
    Atlas::IndexType stored_ptr;
    bool move_forward = true;
    void set_param(const ParamMatrixType params)
    {
        this->param_list = params;
        this->param_ptr = 0;
        this->move_forward = true;
        full_solver_flag = true;
        full_solver_initialize();
    }


    /* GBZ types*/
    typedef Atlas::GBZBaseManifold<_data_GBZ_dim> DataPointViewType;
    typedef Atlas::GBZBaseManifold<_poly_GBZ_dim> PolyPointViewType;
    typedef Eigen::Matrix<typename DataPointViewType::ScalarType, Eigen::Dynamic, DataPointViewType::dim> PointMatrixType;

    const Atlas::ChartIdType n_charts = (0x1<<_data_GBZ_dim);
    virtual bool full_solver_check(std::vector<typename DataPointViewType::PointDataType> curr_sol){
        Atlas::RealType min_dist = Atlas::check_min_distance<DataPointViewType> (curr_sol, critical_distance);
        // STDERR << "min distance" << min_dist << '\n';
        return (min_dist < critical_distance);
    }

    /* initializer */
    aGBZSolver():curr_poly_string_list(DataPointViewType::dim), curr_initial_values(DataPointViewType::dim){}
    // aGBZSolver(){}

    /* Results */
    std::vector<std::vector<typename DataPointViewType::PointDataType>> all_points;

    /* Segmentation and loops */
    std::vector<std::vector<typename PolyPointViewType::PointDataType>> all_segments;
    std::vector<std::vector<typename PolyPointViewType::PointDataType>> all_loops;
    virtual void sol_to_segments() = 0;
    void segments_to_loops(Atlas::RealType glue_tol)
    {
        all_loops.clear();
        std::vector<typename PolyPointViewType::PointDataType> curr_loop = all_segments[all_segments.size() - 1]; 
        all_segments.pop_back();
        while(all_segments.size())
        {
            // STDERR << all_loops.size();
            /* find nearest segment */
            typename std::vector<std::vector<typename PolyPointViewType::PointDataType>>::iterator min_seg_iter = all_segments.begin(); 
            PolyPointViewType viewer1, viewer2;
            viewer1.reinit_data(curr_loop[curr_loop.size()-1].coords.data(), curr_loop[curr_loop.size()-1].chart_labels.data());
            viewer2.reinit_data(((*min_seg_iter)[0]).coords.data(), ((*min_seg_iter)[0]).chart_labels.data());
            Atlas::RealType min_dist = Atlas::distance<PolyPointViewType>(viewer1, viewer2);

            for(auto curr_seg_iter = all_segments.begin(); curr_seg_iter != all_segments.end(); curr_seg_iter ++)
            {
                // STDERR << "hello" << curr_seg_iter->size() << "\n";
                viewer2.reinit_data(((*curr_seg_iter)[0]).coords.data(), ((*curr_seg_iter)[0]).chart_labels.data());
                Atlas::RealType curr_dist = Atlas::distance<PolyPointViewType>(viewer1, viewer2);
                if(curr_dist < min_dist)
                {
                    // STDERR << "How are you\n";
                    min_dist = curr_dist;
                    min_seg_iter = curr_seg_iter;
                }
            }
            if(min_dist>= glue_tol)
            {
                STDERR << "aGBZSolver::segments_to_loops: loop does not close! \n";
                assert(0);
            }
            /* append the new segment to the old one, and erase the segment from the list */
            curr_loop.insert(curr_loop.end(), min_seg_iter->begin(), min_seg_iter->end());
            all_segments.erase(min_seg_iter);

            /* check whether the loop is closed */
            viewer1.reinit_data(curr_loop[curr_loop.size()-1].coords.data(), curr_loop[curr_loop.size()-1].chart_labels.data());
            viewer2.reinit_data(curr_loop[0].coords.data(), curr_loop[0].chart_labels.data());
            if(Atlas::distance(viewer1, viewer2) < glue_tol)
            {
                all_loops.push_back(curr_loop);
                if(all_segments.size())
                {
                    curr_loop = all_segments[all_segments.size() - 1]; 
                    all_segments.pop_back();
                }
            }
            else
            {
                if(! all_segments.size())
                {
                    STDERR << "aGBZSolver::segments_to_loops: loop does not close! \n";
                    assert(0);
                }
            }
        }
    }

    void post_process(Atlas::RealType glue_tol)
    {
        sol_to_segments();
        // STDERR << all_segments.size() << '\n';
        segments_to_loops(glue_tol);
    }

    void loop_readout(Atlas::IndexType loop_id, std::vector<typename PolyPointViewType::ScalarType>& loop_coords, 
        std::vector<Atlas::ChartIdType>& loop_chart_labels)
    {
        Atlas::point_list_to_vec<PolyPointViewType>(all_loops[loop_id], loop_coords, loop_chart_labels);
    }

    void segment_readout(Atlas::IndexType segment_id, std::vector<typename PolyPointViewType::ScalarType> & segment_coords,
            std::vector<Atlas::ChartIdType> & segment_chart_labels)
    {
        Atlas::point_list_to_vec<PolyPointViewType>(all_segments[segment_id], segment_coords, segment_chart_labels);
    }

    void raw_sol_readout(Atlas::IndexType sol_id, std::vector<typename DataPointViewType::ScalarType> & sol_coords,
            std::vector<Atlas::ChartIdType> & sol_chart_labels)
    {
        Atlas::point_list_to_vec<DataPointViewType>(all_points[sol_id], sol_coords, sol_chart_labels);
    }

    /* equation */
    virtual void equation_update() = 0;

    /* solver */
    bool full_solver_flag = true; // 1: full solver; 0: continuation solver
    std::vector<typename DataPointViewType::PointDataType> last_sol, curr_sol;
    virtual bool solution_filter(typename DataPointViewType::PointDataType curr_sol) = 0;

    // For full solvers
    unsigned int solver_curr_chart_binary = 0;

    void full_solver_initialize(){
        solver_curr_chart_binary = 0;
        equation_update();
        curr_sol.clear();
        solver_suspended = false;
        full_solver_update_string();
    }
    std::vector<std::string> curr_poly_string_list;
    bool insert_full_solver_results(const PointMatrixType & full_solver_results, bool no_solution)
    {
        // insert result 
        if(!no_solution)
        {
            for(Atlas::IndexType point_id = 0; point_id < full_solver_results.rows(); point_id++)
            {
                // STDERR << continuation_solver_fun(full_solver_results.row(point_id)) << '\n';
                typename DataPointViewType::PointDataType curr_point;

                binary_to_chart_labels<_data_GBZ_dim>(curr_point.chart_labels, solver_curr_chart_binary);
                
                for(Atlas::IndexType var_id = 0; var_id < DataPointViewType::dim; var_id ++)
                {
                    curr_point.coords[var_id] = full_solver_results(point_id, var_id);
                }
                if(solution_filter(curr_point))
                {
                    curr_sol.push_back(curr_point);
                }
            }
        }

        // prepare for next step
        solver_curr_chart_binary ++;
        if(solver_curr_chart_binary < n_charts)
        {
            equation_update();
            full_solver_update_string();
            return true;
        }
        else
        {
            return false;
        }
    }
    virtual void full_solver_update_string() = 0;

    // For continuation solvers 
    /* return values:
        [
            [fx],
            [jac[0,:]],
            [jac[1,:]],
            ...
        ]
    */
    bool solver_suspended = false;
    void suspend_solver()
    {
        solver_suspended = true;
    }
    Atlas::IndexType continuation_solver_sol_ptr = 0;
    std::vector<typename DataPointViewType::ScalarType> curr_initial_values;
    void continuation_solver_initialize()
    {
        continuation_solver_sol_ptr = 0;
        if(curr_initial_values.size() != DataPointViewType::dim)
        {
            curr_initial_values.clear();
            curr_initial_values.resize(DataPointViewType::dim);
        }
        if(curr_sol.size() != last_sol.size())
        {
            curr_sol.clear();
            curr_sol.resize(last_sol.size());
        }
        // add initial values
        for(Atlas::IndexType var_id = 0; var_id < DataPointViewType::dim; var_id ++)
        {
            curr_initial_values[var_id] = last_sol[continuation_solver_sol_ptr].coords[var_id];
        }
        chart_labels_to_binary<_data_GBZ_dim>(last_sol[continuation_solver_sol_ptr].chart_labels, solver_curr_chart_binary);
        equation_update();
    }
    virtual PointMatrixType continuation_solver_fun(const PointMatrixType & x) = 0;
    bool insert_continuation_solver_results(PointMatrixType continuation_solver_results)
    {
        // STDERR << continuation_solver_fun(continuation_solver_results) << '\n';
        binary_to_chart_labels<_data_GBZ_dim>(curr_sol[continuation_solver_sol_ptr].chart_labels, solver_curr_chart_binary);
        for(Atlas::IndexType var_id = 0; var_id < DataPointViewType::dim; var_id ++)
        {
            curr_sol[continuation_solver_sol_ptr].coords[var_id] = continuation_solver_results(0, var_id);
        }
        if(!solution_filter(curr_sol[continuation_solver_sol_ptr]))
        {
            suspend_solver();
            // STDERR << curr_sol[continuation_solver_sol_ptr].chart_labels[0] << '\n';
            return false;
        }

        // move next
        continuation_solver_sol_ptr ++;
        if(continuation_solver_sol_ptr < last_sol.size())
        {
            // add initial values
            for(Atlas::IndexType var_id = 0; var_id < DataPointViewType::dim; var_id ++)
            {
                curr_initial_values[var_id] = last_sol[continuation_solver_sol_ptr].coords[var_id];
            }
            chart_labels_to_binary<_data_GBZ_dim>(last_sol[continuation_solver_sol_ptr].chart_labels, 
                        solver_curr_chart_binary);
            equation_update();

            return true;
        }
        else
        {
            return false;
        }
    }
    
    // utils
    std::vector<ParamType> get_curr_param()
    {
        std::vector<ParamType> param_vec(param_dim);
        for(unsigned int j = 0; j < param_dim; j++)
        {
            param_vec[j] = param_list(param_ptr, j);
        }
        return param_vec;
    }

    // move next
    bool next()    // if finish, return False
    {

        // STDERR << "aGBZ_solvers.hpp next() line 424: 'last_sol':\n";
        // for(auto point_it = last_sol.begin(); point_it != last_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        // STDERR << "aGBZ_solvers.hpp next() line 437: 'curr_sol':\n";
        // for(auto point_it = curr_sol.begin(); point_it != curr_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        if(solver_suspended)
        {
            // if the solver is suspended, clear the current solution and recalculate by full solver
            full_solver_flag = true;
            full_solver_initialize();
            return true;
        }

        // add new solutions
        if(full_solver_flag)
        {
            std::vector<typename DataPointViewType::PointDataType> curr_unique_sol;
            Atlas::get_unique_points<DataPointViewType>(curr_sol, curr_unique_sol, AGBZ_UNIQUE_TOL);
            if(all_points.size() == 0)
            {
                // first solve
                curr_sol = curr_unique_sol;
            }
            else
            {
                bool match_success = Atlas::match_point_list<DataPointViewType>(last_sol, curr_unique_sol, curr_sol);
                if(! match_success)
                {
                    curr_sol = curr_unique_sol;
                    if(move_forward)
                    {
                        move_forward = false;
                        stored_ptr = param_ptr;
                    }
                    else
                    {
                        stored_ptr = param_ptr;
                    }
                }
            }
        }

        // STDERR << "aGBZ_solvers.hpp next() line 468: 'last_sol':\n";
        // for(auto point_it = last_sol.begin(); point_it != last_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        // STDERR << "aGBZ_solvers.hpp next() line 478: 'curr_sol':\n";
        // for(auto point_it = curr_sol.begin(); point_it != curr_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        // to standard form
        Atlas::batch_to_standard<DataPointViewType>(curr_sol);

        // // DEBUG
        // STDOUT << "curr_sol:\n";
        // for(auto it = curr_sol.begin(); it != curr_sol.end(); it++)
        // {
        //     print_vector((*it).coords);
        //     STDOUT << "\n";
        // }
        if(param_ptr < all_points.size())
        {
            all_points[param_ptr] = curr_sol;
        }
        else
        {
            all_points.push_back(curr_sol);
        }
        // STDOUT << "all_points:\n";
        // for(Atlas::IndexType sol_slice_id = 0; sol_slice_id < all_points.size(); sol_slice_id ++)
        // {
        //     STDOUT << "sol " << sol_slice_id << '\n';
        //     for(auto it = all_points[sol_slice_id].begin(); it != all_points[sol_slice_id].end(); it++)
        //     {
        //         print_vector((*it).coords);
        //         STDOUT << '\n';
        //     }
        // }
        last_sol = curr_sol;
        
        // STDERR << "aGBZ_solvers.hpp next() line 493: 'last_sol':\n";
        // for(auto point_it = last_sol.begin(); point_it != last_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        // STDERR << "aGBZ_solvers.hpp next() line 503: 'curr_sol':\n";
        // for(auto point_it = curr_sol.begin(); point_it != curr_sol.end(); point_it++)
        // {
        //         STDERR << "chart_labels:";
        //         print_vector((*point_it).chart_labels);
        //         STDERR << "coords:";
        //         print_vector((*point_it).coords);
        //         STDERR << '\n';
        // }

        // prepare next solution
        if(move_forward)
        {
            param_ptr ++;
            if(param_ptr < param_list.rows())
            {
                full_solver_flag = full_solver_check(last_sol);

                if(full_solver_flag)
                {
                    full_solver_initialize();
                }
                else
                {
                    continuation_solver_initialize();
                }
                return true;
            }
            else
            {
                return false; // solution finishes
            }
        }
        else
        {
            if(param_ptr > 0)
            {
                param_ptr --;
                full_solver_flag = full_solver_check(last_sol);
                if(full_solver_flag)
                {
                    full_solver_initialize();
                }
                else
                {
                    continuation_solver_initialize();
                }
                return true;
            }
            else
            {
                param_ptr = stored_ptr + 1;
                last_sol = all_points[stored_ptr];
                move_forward = true;
                full_solver_flag = full_solver_check(last_sol);
                if(full_solver_flag)
                {
                    full_solver_initialize();
                }
                else
                {
                    continuation_solver_initialize();
                }
                return true;
            }
        }
    }
};


template<typename Iterable>
size_t get_loop_length(std::vector<Iterable*> loop_list)
{
    size_t total_length = 0;
    for(auto loop_ptr:loop_list)
    {
        total_length += loop_ptr->size();
    }
    return total_length;
}

template<Atlas::IndexType GBZ_dim>
void eval_char_poly_around_loop(CharPolyFun<GBZ_dim> & char_poly, Atlas::IndexType val_id,
     typename Atlas::GBZBaseManifold<GBZ_dim>::PointDataType & curr_point, 
     const std::vector<Atlas::ComplexType> &variable_list, 
     std::vector<Atlas::ComplexType> & val_list){
    /* Warning: the chart_label of "curr_point" and "char_poly" are assumed to be consistent */
        assert(val_list.size() == variable_list.size());

        PolyTools::VarScalarVec point_coords(curr_point.coords.begin(), curr_point.coords.end());
        
        for(Atlas::IndexType variable_id = 0; variable_id < variable_list.size(); variable_id++)
        {
            point_coords[val_id] = variable_list[variable_id];
            val_list[variable_id] = char_poly.get_current_laurant().eval(point_coords);
        }
}

#endif