/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-16
 * @lastModified  2023-07-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "src/atlas/atlas.hpp"
#include "src/atlas/GBZ_manifolds.hpp"
#include <iostream>
#include <math.h>

using namespace Atlas;

int test_distance1()
{
    IndexType N = 1000;
    std::vector<RealN<2>::PointDataType> point_list(N);
    RealType R = 1;
    for(IndexType j = 0; j<N; j++)
    {
        RealType theta = (2.0*M_PI*j)/N;
        point_list[j].coords[0] = cos(theta);
        point_list[j].coords[1] = sin(theta);
    }
    RealType tol = 1e-2;
    RealType min_dist = check_min_distance<RealN<2>>(point_list, tol);
    std::cout <<  min_dist << '\n';
    if(min_dist < tol)
    {
        std::cout << "True!\n";
    }
    else
    {
        std::cout << "False!\n";
    }
    return 0;

}

int test_distance2()
{
    RealType R1 = 0.8, R2 = 0.8;
    CP1::PointDataType P0;
    IndexType N = 1000;
    std::vector<CP1::PointDataType> point_list(2*N);

    P0.chart_labels[0] = 0;
    P0.coords[0] = 0.9;

    for(IndexType j = 0; j < N; j++)
    {
        RealType theta = (2*M_PI*j)/N - M_PI;
        point_list[j].chart_labels[0] = 0;
        point_list[j].coords[0] = std::exp(ComplexType(0, theta)) * R1;
    }
    for(IndexType j = 0; j < N; j++)
    {
        RealType theta = (2*M_PI*j)/N;
        point_list[N + j].chart_labels[0] = 1;
        point_list[N + j].coords[0] = std::exp(ComplexType(0, theta)) * R1;
    }

    IndexType min_id;
    RealType min_dist = calculate_min_distance<CP1>(P0, point_list, min_id);
    std::cout << "Minimal distance: " << min_dist << '\n';
    std::cout << "Index of minimal element: " << min_id << '\n';
    return 0;
}

int test_distance_GBZ_manifold2()
{
    RealType R1 = 0.8, R2 = 0.8;
    GBZBaseManifold<1>::PointDataType P0;
    IndexType N = 1000;
    std::vector<GBZBaseManifold<1>::PointDataType> point_list(2);

    GBZBaseManifold<1> viewer1, viewer2;
    viewer1.generate_new_point(point_list[0], {1, ComplexType(0,0)}, {0});
    viewer2.generate_new_point(point_list[1], {0.5, ComplexType(0.6,0.8)}, {0});
    std::cout << distance(viewer1, viewer2) << '\n';

    return 0;
}

int test_distance_GBZ_manifold()
{
    RealType R1 = 0.8, R2 = 0.8;
    GBZBaseManifold<1>::PointDataType P0;
    IndexType N = 1000;
    std::vector<GBZBaseManifold<1>::PointDataType> point_list(2*N);

    P0.chart_labels[0] = 0;
    P0.coords[0] = 1;
    P0.coords[1] = 0.9;

    for(IndexType j = 0; j < N; j++)
    {
        RealType theta = (2*M_PI*j)/N - M_PI;
        point_list[j].chart_labels[0] = 0;
        point_list[j].coords[1] = std::exp(ComplexType(0, theta)) * R1;
        point_list[j].coords[0] = 1;
    }
    for(IndexType j = 0; j < N; j++)
    {
        RealType theta = (2*M_PI*j)/N;
        point_list[N + j].chart_labels[0] = 1;
        point_list[N + j].coords[1] = std::exp(ComplexType(0, theta)) * R1;
        point_list[N + j].coords[0] = 1;
    }

    IndexType min_id;
    RealType min_dist = calculate_min_distance<GBZBaseManifold<1>>(P0, point_list, min_id);
    std::cout << "Minimal distance: " << min_dist << '\n';
    std::cout << "Index of minimal element: " << min_id << '\n';
    return 0;
}

int main()
{
    // return test_distance2();
    return test_distance_GBZ_manifold();
}