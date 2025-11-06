/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-16
 * @lastModified  2023-07-17
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include <iostream>
#include "atlas.hpp"
#include <complex>

int test_flat()
{
    // RealN<2> point_view1, point_view2;
    // RealN<2>::PointDataType data[2];
    ComplexN<2> point_view1, point_view2;
    ComplexN<2>::PointDataType data[2];
    point_view1.generate_new_point(data[0], {ComplexType(12,16),0}, {});
    point_view2.generate_new_point(data[1], {0, ComplexType(9,12)}, {});
    // std::cout << point_view1.distance(point_view2) << '\n';
    std::cout << distance(point_view1, point_view2) << '\n';
    std::cout << sizeof(point_view1) << '\n';
    std::cout << sizeof(data) << '\n';
    return 0;
}

int test_CP1()
{
    CP1::PointDataType data[2];
    CP1 point_view1, point_view2;
    point_view1.generate_new_point(data[0], {ComplexType(2,0.1)}, {0});
    point_view2.generate_new_point(data[1], {ComplexType(0.6,0.3)}, {1});
    std::cout << distance(point_view1, point_view2) << '\n';

    point_view1.to_chart({1});
    std::cout << distance(point_view1, point_view2) << '\n';
    point_view1.to_chart({1});
    std::cout << distance(point_view1, point_view2) << '\n';
    point_view1.to_chart({0});
    std::cout << distance(point_view1, point_view2) << '\n';
    point_view1.to_chart({1});
    std::cout << distance(point_view1, point_view2) << '\n';
    return 0;
}

int main()
{
    return test_CP1();
}