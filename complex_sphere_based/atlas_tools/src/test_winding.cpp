/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-08-17
 * @lastModified  2023-09-05
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "atlas/GBZ_manifolds.hpp"
#include <eigen3/Eigen/Core>
#include <functional>
#include <fstream>
#include <string>

template<typename Iterator>
void save_iter_to_file(std::string file_name, const Iterator & vec, int with_real_part)
{
    /*
        with_real_part:
            0: real vector
            1: complex vector, output real part
            2: complex vector, output imaginary part
    */
    std::ofstream out_file_pointer(file_name, std::ios::out);
    for(size_t j = 0; j < vec.size() - 1; j ++)
    {
        auto x = vec[j];
        switch (with_real_part)
        {
        case 0:
            out_file_pointer << x << ',';
            break;
        
        case 1:
            out_file_pointer << x.real() << ',';
            break;
        
        case 2:
            out_file_pointer << x.imag() << ',';
            break;
        }
    }
    auto x = vec.back();
    switch (with_real_part)
    {
    case 0:
        out_file_pointer << x ;
        break;
    
    case 1:
        out_file_pointer << x.real() ;
        break;
    
    case 2:
        out_file_pointer << x.imag() ;
        break;
    }
    out_file_pointer.close();
    
}

int test_winding_number()
{
    size_t N = 100;
    std::vector<Atlas::ComplexType> val1(N), val2(N), val3(N), val4(N);
    Eigen::Map<Eigen::VectorXcd> val1_mat(val1.data(), val1.size()), val2_mat(val2.data(), val2.size()), val3_mat(val3.data(), val3.size()), val4_mat(val4.data(), val4.size());
    std::vector<std::vector<Atlas::ComplexType>*> val_ptr_list = {&val1, &val2, &val3, &val4};
    val1_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, 0, M_PI/2)).array());
    val2_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, M_PI/2, M_PI)).array());
    val3_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, M_PI, 3*M_PI/2)).array());
    val4_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, 3*M_PI/2, 2*M_PI)).array());

    int winding = Atlas::winding_number(val_ptr_list);
    std::cout << "Integer winding number:" << winding << '\n';
    return 0;
}

int test_winding_number_iter_range()
{
    size_t N = 100;
    std::vector<Atlas::ComplexType> val1(N), val2(N), val3(N), val4(N);
    Eigen::Map<Eigen::VectorXcd> val1_mat(val1.data(), val1.size()), val2_mat(val2.data(), val2.size()), val3_mat(val3.data(), val3.size()), val4_mat(val4.data(), val4.size());
    std::vector<std::vector<Atlas::ComplexType>*> val_ptr_list = {&val1, &val2, &val3, &val4};
    val1_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, 0, M_PI/2)).array());
    val2_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, M_PI/2, M_PI)).array());
    val3_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, M_PI, 3*M_PI/2)).array());
    val4_mat = Eigen::exp((-Atlas::ComplexType(0,1)*2.0 * Eigen::VectorXcd::LinSpaced(N, 3*M_PI/2, 2*M_PI)).array());

    std::vector<std::array<Atlas::ComplexVec::iterator, 2>> winding_loop = {{val1.begin(), val1.end()},
                                         {val2.begin(), val2.end()},
                                         {val3.begin(), val3.end()},
                                         {val4.begin(), val4.end() - 10}};
    int winding = Atlas::winding_number(winding_loop);
    std::cout << "Integer winding number:" << winding << '\n';
    return 0;
}

const Atlas::ComplexType J(0,1);

namespace Atlas{

void loop_for_point(ComplexType beta, ZeroPointSegment & zero_point, std::vector<ComplexVec*> & val_list, 
    std::vector<std::array<ComplexVec::iterator, 2>> & outer_loop,
    std::vector<std::array<ComplexVec::iterator, 2>> & inner_loop,
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
        ComplexVec* p = new ComplexVec(N_detour);
        p->assign(outer_path.begin(), outer_path.end());
        val_list.push_back(p);
        outer_loop.push_back({p->begin(), p->end()});
        p = new ComplexVec(N_detour);
        p->assign(inner_path.begin(), inner_path.end());
        val_list.push_back(p);
        inner_loop.push_back({p->begin(), p->end()});
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
        
        ComplexVec *p = new ComplexVec(outer_begin.size());
        p->assign(outer_begin.begin(), outer_begin.end());
        val_list.push_back(p);
        outer_loop.push_back({p->begin(), p->end()});

        p = new ComplexVec(outer_arc.size());
        p->assign(outer_arc.begin(), outer_arc.end());
        val_list.push_back(p);
        outer_loop.push_back({p->begin(), p->end()});

        p = new ComplexVec(outer_end.size());
        p->assign(outer_end.begin(), outer_end.end());
        val_list.push_back(p);
        outer_loop.push_back({p->begin(), p->end()});

        p = new ComplexVec(inner_begin.size());
        p -> assign(inner_begin.begin(), inner_begin.end());
        val_list.push_back(p);
        inner_loop.push_back({p->begin(), p->end()});

        p = new ComplexVec(inner_arc.size());
        p -> assign(inner_arc.begin(), inner_arc.end());
        val_list.push_back(p);
        inner_loop.push_back({p->begin(), p->end()});

        p = new ComplexVec(inner_end.size());
        p -> assign(inner_end.begin(), inner_end.end());
        val_list.push_back(p);
        inner_loop.push_back({p->begin(), p->end()});
    }
}


void get_winding_loop(ComplexType beta1, RealType theta_detour, RealType zero_eps,
        IndexType N_points_loop, IndexType N_points_detour, 
        const std::function<void(const ComplexVec &, ComplexVec &)> & fun, 
        ComplexVec & outer_loop, ComplexVec & inner_loop)
{
    const ComplexType J(0,1);

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

    // 2. collect zero points
    bool head_increase = (std::norm(circular_vals[1]) > std::norm(circular_vals[0]));
    bool last_increase = head_increase;
    std::deque<ZeroPointSegment> zero_points;
    zero_points.clear();

    bool tail_increase = (std::norm(circular_vals[N_points_loop-1]) > std::norm(circular_vals[N_points_loop - 2]));
    if(head_increase && (! tail_increase))
    {
        RealType min_id; 
        RealType min_val = get_min_position(
            std::norm(circular_vals[N_points_loop - 2]),
            std::norm(circular_vals[0]),
            std::norm(circular_vals[1]),
            min_id
        );
        if(min_val < zero_eps)
        {
            ZeroPointSegment new_point(min_id);
            new_point.update_indices(indices_detour);
            zero_points.push_back(new_point);
        }
    }

    for(IndexType curr_id = 1; curr_id < N_points_loop-1; curr_id ++)
    {
        bool curr_increase = (std::norm(circular_vals[curr_id + 1]) > std::norm(circular_vals[curr_id]));
        if(curr_increase && (!last_increase))
        {
            RealType min_id;
            RealType min_val = get_min_position(std::norm(circular_vals[curr_id-1]),
                std::norm(circular_vals[curr_id]), 
                std::norm(circular_vals[curr_id + 1]),
                min_id);

            if(min_val < zero_eps)
            {
                ZeroPointSegment new_point(min_id + curr_id);
                new_point.update_indices(indices_detour);
                zero_points.push_back(new_point);
            }
        }
        last_increase = curr_increase;
    }

    // 3. join adjacent bubbles
    if(! zero_points.size())
    {
        STDERR << "Atlas::winding_phase_auto_detour: no zero points found!\n";
        std::runtime_error("");
    }

    std::deque<ZeroPointSegment> new_zero_points;
    new_zero_points.clear();

    // 3.1. get correct head point
    ZeroPointSegment head_point = zero_points.front();
    zero_points.pop_front();

    // 3.1.1. from head
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

    // 3.1.2. from back
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

    // 3.2. traverse the remaining points until exhause zero_points
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

    for(auto curr_point : new_zero_points)
    {
        std::cout << "current point:\n";
        std::cout << "N points:" << curr_point.N_points << '\n';
        std::cout << "left point:" << curr_point.left_point << '\n';
        std::cout << "right point:" << curr_point.right_point << '\n';
        std::cout << "left index:" << curr_point.left_index << '\n';
        std::cout << "right index:" << curr_point.right_index << '\n';
        std::cout << '\n';
    }

    // 4. get winding loop    
    std::vector<ComplexVec *> loops_ptr;
    std::vector<std::array<ComplexVec::iterator,  2>> outer_winding_path, inner_winding_path;

    IndexType begin_ptr, end_ptr;
    ZeroPointSegment & head_point_ref = new_zero_points.front();
    if(head_point_ref.left_index < 0)
    {
        begin_ptr = head_point.right_index; 
        end_ptr = (head_point_ref.left_index + N_points_loop); 
        loop_for_point(beta1, head_point, loops_ptr, outer_winding_path, inner_winding_path, N_points_detour,
            rel_r_detour, theta_detour, N_points_loop, fun);
        new_zero_points.pop_front();
    }
    else
    {
        ZeroPointSegment & tail_point = new_zero_points.back();
        if(tail_point.right_point > (N_points_loop - 1))
        {
            end_ptr = tail_point.left_index + 1;
            begin_ptr = tail_point.right_index + 1 - N_points_loop;
            loop_for_point(beta1, tail_point, loops_ptr, outer_winding_path, inner_winding_path, N_points_detour,
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
            outer_winding_path.push_back({circular_loop.begin() + curr_left_ptr, circular_loop.begin() + curr_right_ptr + 1});
            inner_winding_path.push_back({circular_loop.begin() + curr_left_ptr, circular_loop.begin() + curr_right_ptr + 1});
            loop_for_point(beta1, point, loops_ptr, outer_winding_path, inner_winding_path, 
                N_points_detour, rel_r_detour, theta_detour, N_points_loop, fun);
            curr_left_ptr = point.right_index;
        }
    }
    if(curr_left_ptr < end_ptr)
    {
        outer_winding_path.push_back({circular_loop.begin() + curr_left_ptr, circular_loop.begin() + end_ptr});
        inner_winding_path.push_back({circular_loop.begin() + curr_left_ptr, circular_loop.begin() + end_ptr});
    }

    // 5. collect winding phase
    for(auto ptr_pair:outer_winding_path)
    {
        for(auto curr_it = ptr_pair[0]; curr_it != ptr_pair[1]; curr_it++)
        {
            outer_loop.push_back(*curr_it);
        }
    }

    for(auto ptr_pair:inner_winding_path)
    {
        for(auto curr_it = ptr_pair[0]; curr_it != ptr_pair[1]; curr_it++)
        {
            inner_loop.push_back(*curr_it);
        }
    }

    // 6. clear memory
    for(auto p : loops_ptr)
    {
        delete p;
    }
}

}

class TestFun
{
public:
    std::vector<double> phase_list;
    TestFun(const std::vector<double> & phase_list):phase_list(phase_list){}

    void operator()(const Atlas::ComplexVec & beta_loop, Atlas::ComplexVec & vals)
    {
        for(size_t j = 0; j < beta_loop.size(); j++)
        {
            vals[j] = 1.0;
            for(auto theta : phase_list)
            {
                vals[j] *= (beta_loop[j] - exp(J * theta));
            }
        }
    }

};

int test_zero_point_search()
{
    TestFun test_fun({M_PI/3, M_PI * 4/3, M_PI/3 + 0.01, 2*M_PI/3});
    Atlas::ComplexType beta = exp(J * (M_PI/3.0 + 5e-2));
    Atlas::RealType theta_detour = 5e-3*M_PI;
    int outer_winding, inner_winding;
    // Atlas::ComplexVec outer_loop, inner_loop;
    // Atlas::get_winding_loop(beta, theta_detour, 1e-3, 
    //     200, 50, test_fun, outer_loop, inner_loop);
    Atlas::winding_phase_auto_detour(beta, theta_detour, 1e-3, 
        200, 50, test_fun, outer_winding, inner_winding);
    std::cout << outer_winding << '\n';
    std::cout << inner_winding << '\n';
    
    // save_iter_to_file("data/test_outer_loop_real.csv", outer_loop, 1);
    // save_iter_to_file("data/test_outer_loop_imag.csv", outer_loop, 2);
    // save_iter_to_file("data/test_inner_loop_real.csv", inner_loop, 1);
    // save_iter_to_file("data/test_inner_loop_imag.csv", inner_loop, 2);
    return 0;
}

// int test_self_adaptive()
// {
//     Atlas::ComplexType a = Atlas::ComplexType(1.0 - 1e-3);
//     Atlas::IndexType N = 50;
//     std::function<void(const Atlas::ComplexVec &, Atlas::ComplexVec &)> my_fun = [&a](const Atlas::ComplexVec & beta, Atlas::ComplexVec & val)
//     {
//         for(size_t j = 0; j < beta.size(); j++)
//         {
//             val[j] = 1.0/(beta[j] - a);
//         }
//     };
//     STDOUT << Atlas::winding_phase(0, 2 * M_PI, 1.0, N, 0.1*M_PI, my_fun)<<'\n';
//     return 0;
// }

int main()
{
    // return test_winding_number_iter_range();
    return test_zero_point_search();
}