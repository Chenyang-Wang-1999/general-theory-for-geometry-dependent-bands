/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2024-03-30
 * @lastModified  2024-03-30
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include "src/atlas/atlas.hpp"
#include <vector>
#include <array>

int test_3d_to_CP1()
{
    Atlas::IndexType N_points = 100;
    Atlas::CP1::PointDataType point1, point2;
    Atlas::CP1  viewer1(point1.coords.data(), point1.chart_labels.data()), 
                viewer2(point2.coords.data(), point2.chart_labels.data());
    std::array<Atlas::RealType,3> vec_3d;
    srand(time(NULL));
    for(int j = 0; j < N_points; j ++)
    {
        point1.chart_labels[0] = rand() % 2;
        point1.coords[0] = Atlas::ComplexType(
            rand()/((double) RAND_MAX) - 0.5,
            rand()/((double) RAND_MAX) - 0.5
        );

        viewer1.embed_into_3d(vec_3d);
        viewer2.set_from_3d(vec_3d);
        std::cout << "distance:" << Atlas::distance(viewer1, viewer2) << '\n';
    }
    return 0;
}

int test_midpoint()
{
    Atlas::IndexType N_points = 100;
    Atlas::CP1::PointDataType point1, point2, point3;
    Atlas::CP1  viewer1(point1.coords.data(), point1.chart_labels.data()), 
                viewer2(point2.coords.data(), point2.chart_labels.data()),
                viewer3(point3.coords.data(), point3.chart_labels.data());
    std::array<Atlas::RealType,3> vec_3d;
    srand(time(NULL));
    for(int j = 0; j < N_points; j ++)
    {
        point1.chart_labels[0] = rand() % 2;
        point1.coords[0] = Atlas::ComplexType(
            rand()/((double) RAND_MAX) - 0.5,
            rand()/((double) RAND_MAX) - 0.5
        );

        point2.chart_labels[0] = rand() % 2;
        point2.coords[0] = Atlas::ComplexType(
            rand()/((double) RAND_MAX) - 0.5,
            rand()/((double) RAND_MAX) - 0.5
        );

        viewer1.linear_interp(viewer2, viewer3, 1);
        viewer1.embed_into_3d(vec_3d);
        std::cout << "point 1:";
        for(auto curr_val: vec_3d)
        {
            std::cout << curr_val << ',';
        }
        std::cout << '\n';
        viewer2.embed_into_3d(vec_3d);
        std::cout << "point 2:";
        for(auto curr_val: vec_3d)
        {
            std::cout << curr_val << ',';
        }
        std::cout << '\n';
        viewer3.embed_into_3d(vec_3d);
        std::cout << "point 3:";
        for(auto curr_val: vec_3d)
        {
            std::cout << curr_val << ',';
        }
        std::cout << '\n';
    }
    return 0;   
}

int main()
{
    // return test_3d_to_CP1();
    return test_midpoint();
}