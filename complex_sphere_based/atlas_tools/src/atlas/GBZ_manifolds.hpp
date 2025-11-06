/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-17
 * @lastModified  2025-03-30
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

/*
    Define GBZ related spaces
*/

#ifndef GBZ_MANIFOLDS_HPP
#define GBZ_MANIFOLDS_HPP

#define ATLAS_MAX_RECURSION_DEPTH 20

#include "atlas.hpp"
#include <assert.h>
#include <math.h>
#include <functional>
#include <deque>
// #include "debug_tools.hpp"


namespace Atlas{
typedef std::vector<ComplexType> ComplexVec;


template<IndexType GBZ_dim>
using GBZBaseManifold = AffineDirectProduct<
    ComplexN<1>,
    AffineNProduct<CP1, GBZ_dim>
>;


template<IndexType GBZ_dim>
using CP1Product = AffineNProduct<CP1, GBZ_dim>;


/* generate the winding loop for given beta and exp(iphi) */
#define ATLAS_REINIT_VEC(vec_name, new_size) \
    if(vec_name.size() != new_size)\
    { vec_name.clear(); \
        vec_name.resize(new_size); }
#define ATLAS_MAP_VEC_TO_EIGEN(vec_name, eigen_vec_name) eigen_vec_name(vec_name.data(), vec_name.size())


/* calculate winding number of a vector of complex numbers */
int winding_number(std::vector<ComplexType> loops)
{
    RealType total_winding = 0.0;
    // for(auto it = loops.begin(); it+1 != loops.end(); it++)
    // {
    //     total_winding += std::log((*(it+1))/(*it)).imag();
    // }
    // total_winding += std::log((loops[0])/(*(loops.end()))).imag();
    // total_winding /= (2*M_PI);
    // return total_winding;

    for(IndexType curr_id = 0; curr_id < loops.size()-1; curr_id++)
    {
        total_winding += std::log((loops[curr_id+1])/loops[curr_id]).imag();
    }
    total_winding += std::log((loops[0])/(loops[loops.size()-1])).imag();
    total_winding /= (2*M_PI);
    return lround(total_winding);
}

int winding_number(std::vector<std::vector<ComplexType>*> loop_segments)
{
    RealType total_winding = 0.0;
    for(IndexType segment_ptr_id = 0; segment_ptr_id < loop_segments.size() - 1; segment_ptr_id++)
    {
        for(IndexType point_id = 0; point_id < loop_segments[segment_ptr_id]->size() - 1; point_id++)
        {
            total_winding += std::log((loop_segments[segment_ptr_id]->operator[](point_id+1))/(loop_segments[segment_ptr_id]->operator[](point_id))).imag();
        }
        total_winding += std::log((loop_segments[segment_ptr_id+1]->operator[](0))/(loop_segments[segment_ptr_id]->operator[](loop_segments[segment_ptr_id]->size()-1))).imag();
    }

    for(IndexType point_id = 0; point_id < (*(loop_segments.end()-1)) -> size() - 1; point_id ++)
    {
        total_winding += std::log(((*(loop_segments.end()-1)) -> operator[](point_id + 1))/ ((*(loop_segments.end()-1))->operator[](point_id))).imag();
    }
    total_winding += std::log((loop_segments[0]->operator[](0))/((loop_segments.back()->back()))).imag();
    total_winding /= (2*M_PI);
    // STDOUT << "Winding number float:" << total_winding << '\n';
    return lround(total_winding);
}

int winding_number(std::vector<std::array<ComplexVec::iterator, 2>> & vals)
{
    RealType total_winding = 0.0;
    for(size_t segment_id = 0; segment_id < vals.size()-1; segment_id ++)
    {
        std::array<ComplexVec::iterator, 2> & segment_range = vals[segment_id];
        assert(segment_range[1] != (segment_range[0]));
        for(ComplexVec::iterator curr_it = segment_range[0]; curr_it != segment_range[1] - 1; curr_it++)
        {
            total_winding += std::log((*(curr_it + 1))/(*curr_it)).imag();
        }
        total_winding += std::log((*(vals[segment_id + 1][0]))/(*(segment_range[1] - 1))).imag();
    }

    std::array<ComplexVec::iterator, 2> & segment_range = vals.back();
    assert(segment_range[1] != (segment_range[0]));
    for(ComplexVec::iterator curr_it = segment_range[0]; curr_it != segment_range[1] - 1; curr_it++)
    {
        total_winding += std::log((*(curr_it + 1))/(*curr_it)).imag();
    }
    total_winding += std::log((*(vals.front()[0]))/(*(segment_range[1] - 1))).imag();
    // STDOUT << "total winding number float:" << total_winding/(2 * M_PI) << '\n';
    return lround(total_winding/(2*M_PI));
}

// typedef struct {
//     ComplexType beta1;
//     ComplexType beta2;
//     bool long_bubble = false;
//     IndexType N_detour;
//     IndexType N_arc;

// } PointBubble;

// void join_bubble(PointBubble & bubble_left, PointBubble & bubble_right)
// {
//     if(bubble_left.long_bubble)
//     {
//         if(bubble_right.long_bubble)
//         {
//             bubble_left.beta2 = bubble_right.beta2;
//             bubble_left.N_arc += bubble_right.N_arc + bubble_right.N_detour;
//         }
//         else
//         {
//             bubble_left.beta2 = bubble_right.beta1;
//             bubble_left.N_arc += bubble_right.N_detour;
//         }
//     }
//     else
//     {
//         bubble_left.long_bubble = true;
//         if(bubble_right.long_bubble)
//         {
//             bubble_left.beta2 = bubble_right.beta2;
//             bubble_left.N_detour = bubble_right.N_detour;
//             bubble_left.N_arc = bubble_right.N_arc + bubble_left.N_detour;
//         }
//         else
//         {
//             bubble_left.beta2 = bubble_right.beta1;
//             bubble_left.N_arc = bubble_left.N_detour;
//             bubble_left.N_detour /= 2;
//         }
//     }
// }

/* Generate bubbles for winding loop */
void bubble_gen(ComplexType beta_center, RealType rel_r_detour, RealType remaining_angle, IndexType N_points, 
    ComplexVec & outer_path, ComplexVec & inner_path)
{
    const ComplexType J(0,1);
    // 1. initialize result vectors
    if(outer_path.size() != N_points)
    {
        outer_path.clear();
        outer_path.resize(N_points);
    }
    if(inner_path.size() != N_points)
    {
        inner_path.clear();
        inner_path.resize(N_points);
    }
    Eigen::Map<Eigen::VectorXcd> 
        ATLAS_MAP_VEC_TO_EIGEN(outer_path, outer_path_mat),
        ATLAS_MAP_VEC_TO_EIGEN(inner_path, inner_path_mat);

    // 2. generate outer loop
    ComplexType detour_vec = beta_center * rel_r_detour;
    outer_path_mat = detour_vec * 
            Eigen::exp(J * (Eigen::VectorXd::LinSpaced(N_points, -M_PI/2 - remaining_angle, M_PI/2 + remaining_angle).array())) + beta_center;
    inner_path_mat = - detour_vec * 
            Eigen::exp(J * (Eigen::VectorXd::LinSpaced(N_points, M_PI/2 - remaining_angle, -M_PI/2 + remaining_angle).array())) + beta_center;
}

/* Generate long bubbles for winding loop */
void long_bubble_gen(const std::array<ComplexType, 2> &beta_range, RealType rel_r_detour, RealType remaining_angle, 
    IndexType N_detour, IndexType N_arc, ComplexVec & outer_begin, ComplexVec & inner_begin,
    ComplexVec & outer_arc, ComplexVec & inner_arc, ComplexVec & outer_end, ComplexVec & inner_end)
{
    const ComplexType J(0,1);

    // 1. initialize result vectors
    if(outer_begin.size() != N_detour)
    {
        outer_begin.clear();
        outer_begin.resize(N_detour);
    }
    if(inner_begin.size() != N_detour)
    {
        inner_begin.clear();
        inner_begin.resize(N_detour);
    }
    if(outer_end.size() != N_detour)
    {
        outer_end.clear();
        outer_end.resize(N_detour);
    }
    if(inner_end.size() != N_detour)
    {
        inner_end.clear();
        inner_end.resize(N_detour);
    }
    if(outer_arc.size() != N_arc)
    {
        outer_arc.clear();
        outer_arc.resize(N_arc);
    }
    if(inner_arc.size() != N_arc)
    {
        inner_arc.clear();
        inner_arc.resize(N_arc);
    }

    Eigen::Map<Eigen::VectorXcd>
        ATLAS_MAP_VEC_TO_EIGEN(outer_begin, outer_begin_mat),
        ATLAS_MAP_VEC_TO_EIGEN(inner_begin, inner_begin_mat),
        ATLAS_MAP_VEC_TO_EIGEN(outer_end, outer_end_mat),
        ATLAS_MAP_VEC_TO_EIGEN(inner_end, inner_end_mat),
        ATLAS_MAP_VEC_TO_EIGEN(outer_arc, outer_arc_mat),
        ATLAS_MAP_VEC_TO_EIGEN(inner_arc, inner_arc_mat);

    // 2. prepare parameters
    ComplexType start_out_pointing_vec = beta_range[0] * rel_r_detour;
    ComplexType end_out_pointing_vec = beta_range[1] * rel_r_detour;
    RealType arc_phase = std::log(beta_range[1]/beta_range[0]).imag();
    if(arc_phase < 0)
    {
        arc_phase += 2 * M_PI;
    }

    // 3. evaluate on each path
    outer_begin_mat = start_out_pointing_vec * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_detour, - M_PI/2 - remaining_angle, 0).array()) + beta_range[0];
    inner_begin_mat = - start_out_pointing_vec * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_detour, M_PI/2 - remaining_angle, 0).array()) + beta_range[0];
    outer_arc_mat = beta_range[0] * (1 + rel_r_detour) * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_arc, 0, arc_phase).array());
    inner_arc_mat = beta_range[0] * (1 - rel_r_detour) * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_arc, 0, arc_phase).array());
    outer_end_mat = end_out_pointing_vec * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_detour, 0, M_PI/2 + remaining_angle).array()) + beta_range[1];
    inner_end_mat = - end_out_pointing_vec * Eigen::exp(J * Eigen::VectorXd::LinSpaced(N_detour, 0, -M_PI/2 + remaining_angle).array()) + beta_range[1];
    
}

RealType get_min_position(RealType A, RealType B, RealType C, RealType & min_id)
{
    if(abs(A+C-2*B) < 1e-10)
    {
        min_id = 0;
        return B;
    }
    else
    {
        min_id = (A - C) / (2 * (A + C - 2 * B));
        return (B - (C - A) * (C - A)/(8 * (A + C - 2 * B)));
    }
}

struct ZeroPointSegment
{
    RealType left_point;
    RealType right_point;
    size_t N_points;
    int left_index;
    int right_index;
    ZeroPointSegment(RealType point){
        left_point = point;
        right_point = point;
        N_points = 1;
    }
    ZeroPointSegment(RealType left_point, RealType right_point, size_t N_points):
    left_point(left_point), right_point(right_point), N_points(N_points){}

    void update_indices(RealType indices_detour)
    {
        left_index = (int)(floor(left_point - indices_detour));
        right_index = (int)(ceil(right_point + indices_detour));
    }
};

void eval_for_point(ComplexType beta, ZeroPointSegment & zero_point, std::vector<ComplexVec> & val_data_list, 
    std::vector<std::array<ComplexVec::iterator, 2>> & vals_outer_loop,
    std::vector<std::array<ComplexVec::iterator, 2>> & vals_inner_loop,
    IndexType N_detour, RealType rel_r_detour, RealType remaining_phase,
    IndexType N_loop,
    std::function<void(const ComplexVec &, ComplexVec &)> fun)
{
    const ComplexType J(0,1);
    if(zero_point.N_points == 1)
    {
        ComplexVec outer_path, inner_path;
        ComplexType beta_center = beta * exp(J * 2.0 * M_PI*(zero_point.left_point)/(N_loop-1.0));
        bubble_gen(beta_center, rel_r_detour, remaining_phase, N_detour, outer_path, inner_path);
        val_data_list.push_back(ComplexVec(N_detour));
        fun(outer_path, val_data_list.back());
        vals_outer_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(N_detour));
        fun(inner_path, val_data_list.back());
        vals_inner_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});
    }
    else
    {
        ComplexVec outer_begin, outer_arc, outer_end, inner_begin, inner_arc, inner_end;
        std::array<ComplexType, 2> beta_pair = {
            beta * exp(J * 2.0 * M_PI * (zero_point.left_point)/(N_loop - 1.0)),
            beta * exp(J * 2.0 * M_PI * (zero_point.right_point)/(N_loop - 1.0))
        };
        long_bubble_gen(beta_pair, rel_r_detour, remaining_phase,
            N_detour/2, N_detour * (zero_point.N_points - 1), outer_begin,
            inner_begin, outer_arc, inner_arc, outer_end, inner_end);
        
        val_data_list.push_back(ComplexVec(outer_begin.size()));
        fun(outer_begin, val_data_list.back());
        vals_outer_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(outer_arc.size()));
        fun(outer_arc, val_data_list.back());
        vals_outer_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(outer_end.size()));
        fun(outer_end, val_data_list.back());
        vals_outer_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(inner_begin.size()));
        fun(inner_begin, val_data_list.back());
        vals_inner_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(inner_arc.size()));
        fun(inner_arc, val_data_list.back());
        vals_inner_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});

        val_data_list.push_back(ComplexVec(inner_end.size()));
        fun(inner_end, val_data_list.back());
        vals_inner_loop.push_back({val_data_list.back().begin(), val_data_list.back().end()});
    }
}

/* collect zero points for auto detour*/
bool collect_zero_point(const ComplexVec & circular_vals, IndexType N_points_loop, 
    RealType zero_eps, RealType indices_detour, std::deque<ZeroPointSegment> & new_zero_points)
{
    // 0. get modulus list and the max value
    std::vector<RealType> circular_vals_norm(circular_vals.size());
    RealType max_norm = 0.0;
    auto circular_norm_it = circular_vals_norm.begin();
    for(auto val:circular_vals)
    {
        RealType curr_norm = std::norm(val);
        (*circular_norm_it) = curr_norm;
        if(curr_norm > max_norm)
        {
            max_norm = curr_norm;
        }
        circular_norm_it ++;
    }

    // 1. collect zero points
    bool head_increase = (circular_vals_norm[1] > circular_vals_norm[0]);
    std::deque<ZeroPointSegment> zero_points;
    zero_points.clear();

    RealType global_min_val = 1e10;
    bool tail_increase = (circular_vals_norm[N_points_loop-1] > circular_vals_norm[N_points_loop - 2]);
    if(head_increase && (! tail_increase))
    {
        RealType min_id; 
        RealType min_val = get_min_position(
            circular_vals_norm[N_points_loop - 2],
            circular_vals_norm[0],
            circular_vals_norm[1],
            min_id
        );
        if(min_val < global_min_val)
        {
            global_min_val = min_val;
        }
        if(min_val < zero_eps*max_norm)
        {
            ZeroPointSegment new_point(min_id);
            new_point.update_indices(indices_detour);
            zero_points.push_back(new_point);
        }
    }

    bool last_increase = (circular_vals_norm[1] > circular_vals_norm[0]);
    for(IndexType curr_id = 1; curr_id < N_points_loop-1; curr_id ++)
    {
        bool curr_increase = circular_vals_norm[curr_id + 1] > circular_vals_norm[curr_id];
        if(curr_increase && (!last_increase))
        {
            RealType min_id;
            RealType min_val = get_min_position(circular_vals_norm[curr_id-1],
                circular_vals_norm[curr_id], 
                circular_vals_norm[curr_id + 1],
                min_id);

            if(min_val < global_min_val)
            {
                global_min_val = min_val;
            }
            if(min_val < zero_eps*max_norm)
            {
                ZeroPointSegment new_point(min_id + curr_id);
                new_point.update_indices(indices_detour);
                zero_points.push_back(new_point);
            }
        }
        last_increase = curr_increase;
    }

    // 2. join adjacent bubbles
    if(! zero_points.size())
    {
        // for(auto x:circular_vals)
        // {
        //     STDERR << std::norm(x) << ',';
        // }
        // STDERR << '\n';
        if(global_min_val < 1e10)
        {
            STDERR << "Warning: Atlas::winding_phase_auto_detour: no zero points found!\n";
            STDERR << "min value:" << global_min_val << '\n';
        }
        // assert(0);
        return false;
    }

    if(new_zero_points.size())
    {
        new_zero_points.clear();
    }

    // 2.1. get correct head point
    ZeroPointSegment head_point = zero_points.front();
    zero_points.pop_front();

    // 2.1.1. from head
    if(zero_points.size())
    {
        while(zero_points.front().left_index <= head_point.right_index)
        {
            head_point.right_point = zero_points.front().right_point;
            head_point.N_points ++;
            head_point.update_indices(indices_detour);
            zero_points.pop_front();
            if(!zero_points.size())
            {
                break;
            }
        }
    }

    // 2.1.2. from back
    if(zero_points.size())
    {
        while(zero_points.back().right_index >= (head_point.left_index + (int)(N_points_loop - 1)))
        {
            head_point.left_point = zero_points.back().left_point - (int)(N_points_loop - 1);
            head_point.N_points ++;
            head_point.update_indices(indices_detour);
            zero_points.pop_back();
            if(!zero_points.size())
            {
                break;
            }
        }
    }
    new_zero_points.push_back(head_point);

    // 2.2. traverse the remaining points until exhause zero_points
    while(zero_points.size())
    {
        ZeroPointSegment curr_zero_point = zero_points.front();
        zero_points.pop_front();
        if(zero_points.size())
        {
            while(curr_zero_point.right_index >= (zero_points.front().left_index))
            {
                curr_zero_point.right_point = zero_points.front().right_point;
                curr_zero_point.N_points ++;
                curr_zero_point.update_indices(indices_detour);
                zero_points.pop_front();
                if(!zero_points.size())
                {
                    break;
                }
            }
        }
        new_zero_points.push_back(curr_zero_point);
    }
    return true;
}

/* Calculate winding phase by automatic detour */
bool winding_phase_auto_detour(ComplexType beta1, RealType theta_detour, RealType zero_eps,
        IndexType N_points_loop, IndexType N_points_detour, 
        const std::function<void(const ComplexVec &, ComplexVec &)> & fun, 
        int & outer_winding, int & inner_winding)
{
    const ComplexType J(0,1);

    // STDERR << beta1 << '\n';

    // 0. parameters
    RealType rel_r_detour = 2 * sin(theta_detour);
    RealType indices_detour = theta_detour/(M_PI/(N_points_loop-1));

    // 1. generate circular winding loop
    ComplexVec circular_loop(N_points_loop);
    Eigen::VectorXd phase_list = Eigen::VectorXd::LinSpaced(N_points_loop, 0, 2 * M_PI);
    Eigen::Map<Eigen::VectorXcd> ATLAS_MAP_VEC_TO_EIGEN(circular_loop, circular_loop_mat);
    circular_loop_mat = beta1 * Eigen::exp(J * phase_list.array());
    ComplexVec circular_vals(N_points_loop);
    fun(circular_loop, circular_vals);

    // 2. get zero points
    std::deque<ZeroPointSegment> new_zero_points;
    bool has_zero_point = collect_zero_point(circular_vals, N_points_loop, zero_eps, indices_detour, new_zero_points);
    if(! has_zero_point)
    {
        return false;
    }

    // IndexType point_counter = 0;
    // for(auto point: new_zero_points)
    // {
    //     STDERR << "zero point " << point_counter << '\n';
    //     STDERR << "N points:" << point.N_points << '\n';
    //     STDERR << "left point:" << point.left_point << '\n';
    //     STDERR << "right point:" << point.right_point << '\n';
    //     STDERR << "left index:" << point.left_index << '\n';
    //     STDERR << "right index:" << point.right_index << '\n';
    //     point_counter ++;
    // }

    // 3. get winding values     
    std::vector<ComplexVec> val_data_list;
    std::vector<std::array<ComplexVec::iterator,  2>> outer_winding_vals, inner_winding_vals;

    IndexType begin_ptr, end_ptr;
    ZeroPointSegment & head_point_ref = new_zero_points.front();
    if(head_point_ref.left_index < 0)
    {
        begin_ptr = head_point_ref.right_index; 
        end_ptr = (head_point_ref.left_index + N_points_loop); 
        eval_for_point(beta1, head_point_ref, val_data_list, outer_winding_vals, inner_winding_vals, N_points_detour,
            rel_r_detour, theta_detour, N_points_loop, fun);
        new_zero_points.pop_front();
    }
    else
    {
        ZeroPointSegment & tail_point = new_zero_points.back();
        if(tail_point.right_index > (int)(N_points_loop - 1))
        {
            end_ptr = tail_point.left_index + 1;
            begin_ptr = tail_point.right_index + 1 - N_points_loop;
            eval_for_point(beta1, tail_point, val_data_list, outer_winding_vals, inner_winding_vals, N_points_detour,
            rel_r_detour, theta_detour, N_points_loop, fun);
            new_zero_points.pop_back();
        }
        else
        {
            end_ptr = N_points_loop;
            begin_ptr = 0;
        }
    }


    IndexType curr_left_ptr = begin_ptr;
    if(new_zero_points.size())
    {
        for(auto point: new_zero_points)
        {
            IndexType curr_right_ptr = point.left_index;
            outer_winding_vals.push_back({circular_vals.begin() + curr_left_ptr, circular_vals.begin() + curr_right_ptr + 1});
            inner_winding_vals.push_back({circular_vals.begin() + curr_left_ptr, circular_vals.begin() + curr_right_ptr + 1});
            eval_for_point(beta1, point, val_data_list, outer_winding_vals, inner_winding_vals, 
                N_points_detour, rel_r_detour, theta_detour, N_points_loop, fun);
            curr_left_ptr = point.right_index;
        }
    }
    if(curr_left_ptr < end_ptr)
    {
        outer_winding_vals.push_back({circular_vals.begin() + curr_left_ptr, circular_vals.begin() + end_ptr});
        inner_winding_vals.push_back({circular_vals.begin() + curr_left_ptr, circular_vals.begin() + end_ptr});
    }

    // 4. collect winding phase
    outer_winding = winding_number(outer_winding_vals);
    inner_winding = winding_number(inner_winding_vals);
    // STDERR << outer_winding << ','<< inner_winding << '\n';
    // if(outer_winding - inner_winding < 2)
    // {
    //     save_iter_to_file("data/winding_vals_real.csv", circular_vals, 1);
    //     save_iter_to_file("data/winding_vals_imag.csv", circular_vals, 2);
    //     STDERR << "zero_eps:" << zero_eps << '\n';
    //     STDERR << "indices_detour:" << indices_detour << '\n';
    //     STDERR << "N_loop_points:" << N_points_loop << '\n';
    //     STDERR << outer_winding << ','<< inner_winding << '\n';
    //     assert(0);
    // }

    return true;
}

/* Generate winding loop for GBZ test */
void winding_loop_gen(ComplexType beta1, ComplexType expiphi, 
    RealType rel_r_detour, IndexType N_points_loop, IndexType N_points_detour,
    std::vector<ComplexType> & path_12, 
    std::vector<ComplexType> & circ_2,
    std::vector<ComplexType> & inner_circ_2,
    std::vector<ComplexType> & path_21,
    std::vector<ComplexType> & circ_1, 
    std::vector<ComplexType> & inner_circ_1)
    {
        // initialize all lists
        ATLAS_REINIT_VEC(path_12, N_points_loop)
        ATLAS_REINIT_VEC(circ_2, N_points_detour)
        ATLAS_REINIT_VEC(inner_circ_2, N_points_detour)
        ATLAS_REINIT_VEC(path_21, N_points_loop)
        ATLAS_REINIT_VEC(circ_1, N_points_detour)
        ATLAS_REINIT_VEC(inner_circ_1, N_points_detour)

        // define eigen matrix on the results
        Eigen::Map<Eigen::VectorXcd> 
            ATLAS_MAP_VEC_TO_EIGEN(path_12, path_12_mat), 
            ATLAS_MAP_VEC_TO_EIGEN(circ_2, circ_2_mat), 
            ATLAS_MAP_VEC_TO_EIGEN(inner_circ_2, inner_circ_2_mat), 
            ATLAS_MAP_VEC_TO_EIGEN(path_21, path_21_mat), 
            ATLAS_MAP_VEC_TO_EIGEN(circ_1, circ_1_mat), 
            ATLAS_MAP_VEC_TO_EIGEN(inner_circ_1, inner_circ_1_mat)
        ;

        // generate each path
        RealType phi_angle = std::log(-expiphi).imag() + M_PI;
        ComplexType beta2 = beta1 * expiphi;
        RealType beta_norm = abs(beta1);
        RealType r_detour = rel_r_detour * beta_norm;
        RealType theta = 2 * asin(rel_r_detour/2);
        RealType beta1_arg = log(beta1).imag();
        RealType beta2_arg = beta1_arg + phi_angle;

        Eigen::VectorXd 
            arg_path_12 = Eigen::VectorXd::LinSpaced(N_points_loop, beta1_arg + theta, beta2_arg - theta), 
            arg_path_21 = Eigen::VectorXd::LinSpaced(N_points_loop, beta2_arg + theta , beta1_arg + 2*M_PI - theta), 
            arg_circ_1 = Eigen::VectorXd::LinSpaced(N_points_detour, beta1_arg - M_PI/2 - theta/2, beta1_arg + M_PI/2 + theta/2),
            arg_circ_2 = Eigen::VectorXd::LinSpaced(N_points_detour, beta2_arg - M_PI/2 - theta/2, beta2_arg + M_PI/2 + theta/2),
            arg_inner_1 = Eigen::VectorXd::LinSpaced(N_points_detour, beta1_arg + 3 * M_PI/2 - theta/2, beta1_arg + M_PI/2 + theta/2),
            arg_inner_2 = Eigen::VectorXd::LinSpaced(N_points_detour, beta2_arg + 3 * M_PI/2 - theta/2, beta2_arg + M_PI/2 + theta/2);
        
        const ComplexType J(0,1);
        path_12_mat = Eigen::exp((J*arg_path_12).array()) * beta_norm;
        path_21_mat = Eigen::exp((J*arg_path_21).array()) * beta_norm;
        circ_1_mat = Eigen::exp((J*arg_circ_1).array()) * r_detour + beta1;
        circ_2_mat = Eigen::exp((J*arg_circ_2).array()) * r_detour + beta2;
        inner_circ_1_mat = Eigen::exp((J*arg_inner_1).array()) * r_detour + beta1;
        inner_circ_2_mat = Eigen::exp((J*arg_inner_2).array()) * r_detour + beta2;
    }

/* generate winding loop with radius |beta1|*(1 + rel_r_detour) and |beta1|*(1 - rel_r_detour) */
void winding_loop_gen_circ(ComplexType beta1, RealType rel_r_detour, IndexType N_points_loop,
    std::vector<ComplexType> & outer_path, std::vector<ComplexType> & inner_path)
    {
        const ComplexType J = ComplexType(0.0, 1.0);
        if(outer_path.size() != N_points_loop)
        {
            outer_path.clear();
            outer_path.resize(N_points_loop);
        }
        if(inner_path.size() != N_points_loop)
        {
            inner_path.clear();
            inner_path.resize(N_points_loop);
        }

        // beta0
        ComplexType outer_beta0 = beta1 * (1 + rel_r_detour);
        ComplexType inner_beta0 = beta1 * (1 - rel_r_detour);

        // map to eigen
        Eigen::Map<Eigen::VectorXcd> 
                ATLAS_MAP_VEC_TO_EIGEN(outer_path, mat_outer_path),
                ATLAS_MAP_VEC_TO_EIGEN(inner_path, mat_inner_path);
        
        // generate path
        mat_inner_path = Eigen::exp(J * (Eigen::VectorXd::LinSpaced(N_points_loop, 0, 2 * M_PI).array()))*inner_beta0;
        mat_outer_path = Eigen::exp(J * (Eigen::VectorXd::LinSpaced(N_points_loop, 0, 2 * M_PI).array()))*outer_beta0;

    }

/* 
    Self-adapted winding number calculator for circular loops
        generate a loop from start_phase to end_phase, with given radius
*/
double winding_phase(RealType start_phase, RealType end_phase, RealType radius, IndexType N_points, 
        RealType dphase_th, const std::function<void(const ComplexVec&, ComplexVec&)>& fun, 
        bool & zero_point_flag, size_t recursion_depth = 0){
    const ComplexType J = ComplexType(0.0, 1.0);
    if(end_phase <= start_phase)
    {
        end_phase += 2 * M_PI;
    }

    // STDERR << "From:" << start_phase << ", to:"<<end_phase << ", N_points:" << N_points << '\n';
    if(recursion_depth >= ATLAS_MAX_RECURSION_DEPTH)
    {
        zero_point_flag = false;
        return 0.0;
    }

    // 1. generate the points
    ComplexVec beta_path(N_points);
    Eigen::Map<Eigen::VectorXcd> beta_path_mat(beta_path.data(), beta_path.size());
    Eigen::VectorXd phase_vec = Eigen::VectorXd::LinSpaced(N_points, start_phase, end_phase);
    beta_path_mat = Eigen::exp(J * (phase_vec.array())) * radius;

    // 2. evaluate along these points
    ComplexVec vals(N_points);
    fun(beta_path, vals);

    // print_vector(vals);
    // STDERR << "\n";

    // 3. calculate winding phase
    RealType dphase;
    RealType total_phase = 0.0;
    for(IndexType point_id = 0; point_id < beta_path.size() - 1; point_id++)
    {
        dphase = log((vals[point_id + 1]/ vals[point_id])).imag();
        if(abs(dphase) < dphase_th)
        {
            total_phase += dphase;
        }
        else
        {
            // STDERR << "radius:" << radius << '\n';
            // STDERR << "current dphase:" << dphase << '\n';
            // print_vector(vals);
            // STDERR << '\n';
            IndexType N_split = lround(abs(dphase/dphase_th)) + 3;
            total_phase += winding_phase(phase_vec(point_id), phase_vec(point_id + 1), radius, N_split, dphase_th, fun, zero_point_flag, recursion_depth + 1);
            if(!zero_point_flag)
            {
                return 0.0;
            }
        }
    }
    return total_phase;

}

template<typename ParamType>
double winding_phase(RealType start_phase, RealType end_phase, RealType radius, IndexType N_points, 
        RealType dphase_th, ParamType start_param, const std::function<void(const ComplexVec&, ComplexVec&, typename std::vector<ParamType> &, ParamType)>& fun){
    const ComplexType J = ComplexType(0.0, 1.0);
    if(end_phase <= start_phase)
    {
        end_phase += 2 * M_PI;
    }

    // 1. generate the points
    ComplexVec beta_path(N_points);
    Eigen::Map<Eigen::VectorXcd> beta_path_mat(beta_path.data(), beta_path.size());
    Eigen::VectorXd phase_vec = Eigen::VectorXd::LinSpaced(N_points, start_phase, end_phase);
    beta_path_mat = Eigen::exp(J * (phase_vec.array())) * radius;

    // 2. evaluate along these points and params
    ComplexVec vals(N_points);
    typename std::vector<ParamType> param_list(N_points);
    fun(beta_path, vals, param_list, start_param);

    // 3. calculate winding phase
    RealType dphase;
    RealType total_phase = 0.0;
    for(IndexType point_id = 0; point_id < beta_path.size() - 1; point_id++)
    {
        dphase = log((vals[point_id + 1]/ vals[point_id])).imag();
        if(abs(dphase) < dphase_th)
        {
            total_phase += dphase;
        }
        else
        {
            IndexType N_split = lround(abs(dphase)/dphase_th) + 3;
            total_phase += winding_phase(phase_vec(point_id), phase_vec(point_id + 1), radius, N_split, dphase_th, param_list[point_id], fun);
        }
    }
    return total_phase;

}



}



#endif