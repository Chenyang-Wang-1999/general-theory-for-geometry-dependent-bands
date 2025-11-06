/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-25
 * @lastModified  2025-03-31
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "atlas/atlas.hpp"
#include "atlas/GBZ_manifolds.hpp"
#include <random>
#include <iostream>

template<typename Iterable>
void print_vector(Iterable vector)
{
    for(auto it = vector.begin(); it!= vector.end(); it++)
    {
        std::cout << (*it) << ',';
    }
}

int test_get_submanifold()
{
    static const Atlas::IndexType GBZ_dim = 5;
    Atlas::GBZBaseManifold<GBZ_dim> GBZ_viewer;
    std::vector<Atlas::GBZBaseManifold<GBZ_dim>::PointDataType> GBZ_data_list(20);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0, 1);
    std::uniform_int_distribution<Atlas::ChartIdType> uniform_int(0,1);

    // set data
    for(size_t j = 0; j < 20; j ++)
    {
        GBZ_viewer.reinit_data(GBZ_data_list[j].coords.data(), 
                GBZ_data_list[j].chart_labels.data());
        GBZ_viewer.get_left().set_data({Atlas::ComplexType(normal_dist(gen), normal_dist(gen))}, {});
        for(size_t k = 0; k < GBZ_dim; k++)
        {
            std::cout << uniform_int(gen) << '\n';
            GBZ_viewer.get_right().get(k).set_data({Atlas::ComplexType(normal_dist(gen), normal_dist(gen))}, {uniform_int(gen)});
        }
    }

    // print from list
    std::cout << "Print from list\n";
    for(size_t j = 0; j < 20; j++)
    {
        std::cout << "Chart labels:";
        print_vector(GBZ_data_list[j].chart_labels);
        std::cout << " . Coordinates:";
        print_vector(GBZ_data_list[j].coords);
        std::cout << "\n";
    }

    // print from get method
    for(size_t j = 0; j < 20; j ++)
    {
        GBZ_viewer.reinit_data(GBZ_data_list[j].coords.data(), 
                GBZ_data_list[j].chart_labels.data());
        Atlas::ComplexN<1>::PointDataType curr_data;
        GBZ_viewer.get_left().copy_data(curr_data);
        std::cout << "E part: chart labels ";
        print_vector(curr_data.chart_labels);
        std::cout << " coordinates: ";
        print_vector(curr_data.coords);

        std::cout << "beta part: \n";
        for(size_t k = 0; k < GBZ_dim; k++)
        {
            Atlas::CP1::PointDataType curr_CP1_data;
            GBZ_viewer.get_right().get(k).copy_data(curr_CP1_data);
            std::cout << "chart labels ";
            print_vector(curr_CP1_data.chart_labels);
            std::cout << ", coordinates ";
            print_vector(curr_CP1_data.coords);
            std::cout << '\n';
        }
    }


    return 0;
}

int test_GBZ_manifold_distance()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0, 1);
    std::uniform_int_distribution<Atlas::ChartIdType> binary_dist(0,1);
    std::uniform_real_distribution<double> theta_dist(0, M_PI), phi_dist(0, 2 * M_PI);

    // generate points
    std::array<double, 3> p[6];
    Atlas::ComplexType E[2];

    Atlas::GBZBaseManifold<3> viewer, viewer2;
    Atlas::GBZBaseManifold<3>::PointDataType point[2];
    for(int j = 0; j<2 ; j++)
    {
        viewer.reinit_data(point[j].coords.data(), point[j].chart_labels.data());
        E[j] = Atlas::ComplexType(normal_dist(gen), normal_dist(gen));
        viewer.get_left().set_data({E[j]},{});
        for(int k = 0; k < 3; k++)
        {
            Atlas::RealType theta = theta_dist(gen), phi = phi_dist(gen);
            p[3*j + k] = {sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)};
            viewer.get_right().get(k).set_data({tan((M_PI- theta)/2) * std::exp(Atlas::ComplexType(0,1)*phi)}, {0});
            viewer.get_right().get(k).to_chart({binary_dist(gen)});
        }
    }

    // calculate distance
    Atlas::RealType dist_sq_theo = 0;
    dist_sq_theo = ((E[0] - E[1]) * (std::conj(E[0]- E[1]))).real();
    for(int j = 0 ; j<3; j++)
    {
        for(int k = 0; k<3; k++)
        {
            dist_sq_theo += pow(p[j][k] - p[3+j][k], 2);
        }
    }
    print_vector(point[0].chart_labels);
    std::cout << '\n';
    print_vector(point[1].chart_labels);
    std::cout << '\n';

    std::cout << "Theoretical distance:" << sqrt(dist_sq_theo) << '\n';

    viewer.reinit_data(point[0].coords.data(), point[0].chart_labels.data());
    viewer2.reinit_data(point[1].coords.data(), point[1].chart_labels.data());
    std::cout << "Atlas method:" << distance(viewer, viewer2) <<'\n';

    return 0;
}

int main()
{
    return test_get_submanifold();
    // return test_GBZ_manifold_distance();
}