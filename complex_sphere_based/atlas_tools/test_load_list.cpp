/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-08-15
 * @lastModified  2023-08-15
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "src/atlas/atlas.hpp"
#include "src/atlas/GBZ_manifolds.hpp"

using namespace Atlas;

int main()
{
    IndexType N = 1000;
    std::list<GBZBaseManifold<1>::PointDataType> point_list;
    for(IndexType j = 0; j<N; j++)
    {
        RealType theta = (2.0*M_PI*j)/N;
        GBZBaseManifold<1>::PointDataType curr_point;
        curr_point.chart_labels[0] = j % 2;
        curr_point.coords[1] = ComplexType(cos(theta), sin(theta));
        curr_point.coords[0] = 1.0;
        point_list.push_back(curr_point);
    }

    std::vector<ComplexType> coords;
    std::vector<ChartIdType> chart_labels;
    Atlas::point_list_to_vec<GBZBaseManifold<1>>(point_list, coords, chart_labels);

    /* Output */
    std::cout << "Coordinates:\n";
    for(auto item:coords)
    {
        std::cout << item << ',';
    }
    std::cout << '\n';

    std::cout << "Chart Labels:\n";
    for(auto item:chart_labels)
    {
        std::cout << item << ',';
    }
    std::cout << '\n';
}