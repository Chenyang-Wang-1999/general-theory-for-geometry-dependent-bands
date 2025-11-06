/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-25
 * @lastModified  2023-08-31
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "aGBZ_solvers.hpp"
#include <random>
#include <bitset>

// template<typename Iterable>
// void print_vector(Iterable vector)
// {
//     for(auto it = vector.begin(); it!= vector.end(); it++)
//     {
//         std::cout << (*it) << ',';
//     }
// }

int test_conversion()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<Atlas::ChartIdType> uniform_int(0,1);

    const unsigned int GBZ_dim = 3;

    std::array<Atlas::ChartIdType, GBZ_dim> chart_labels;
    unsigned int chart_labels_int;
    for(auto it = chart_labels.begin(); it != chart_labels.end(); ++it)
    {
        (*it) = uniform_int(gen);
    }

    std::cout << "Chart labels to binary: \n";
    print_vector(chart_labels);
    std::cout << '\n';
    chart_labels_to_binary<GBZ_dim>(chart_labels, chart_labels_int);
    std::cout << std::bitset<GBZ_dim>(chart_labels_int) << '\n';

    std::cout << "Binary to chart labels: \n";
    chart_labels_int = rd() % (0x1 << GBZ_dim);
    binary_to_chart_labels<GBZ_dim>(chart_labels, chart_labels_int);
    print_vector(chart_labels);
    std::cout << '\n';
    std::cout << std::bitset<GBZ_dim>(chart_labels_int) << '\n';

    return 0;
}

int test_polynomial_initialization()
{
    const Atlas::IndexType GBZ_dim = 3;
    PolyTools::Laurant char_poly(GBZ_dim + 1);

    /* set char poly */
    std::vector<int> orders = {
        0, 1, 0, 0,
        0, 1, -2, 0,
        1, 1, -1, -1,
        2, 0, 0, -2,
        0, 0, 0, -1,
        1, 0, 0, 0,
        2, -2, 0, 0,
        0, 0, -2, 0,
        0, -1, 0, 0,
        0, 0, 0, 1
    };
    PolyTools::ScalarVec coeffs = {
        1,2,3,4,5,6,7,8,9,10
    };
    char_poly.set_Laurant_by_terms(coeffs, orders);

    /* set function */
    CharPolyFun<GBZ_dim> char_poly_fun(char_poly);

    /* print functions */
    for(unsigned int chart_labels_int = 0; chart_labels_int < 0x1<<GBZ_dim; chart_labels_int ++)
    {
        std::cout << "chart labels: " << std::bitset<GBZ_dim>(chart_labels_int) << '\n';
        std::cout << char_poly_fun.char_poly_list[chart_labels_int]->num.to_str({"E", "x", "y", "z"}) << '\n';
        print_vector(char_poly_fun.char_poly_list[chart_labels_int]->denom_orders);
        std::cout << "\n\n";
    }
    return 0;
}

int test_evaluation()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<Atlas::ChartIdType> uniform_int(0,1);
    std::normal_distribution<double> normal_dist(0, 1);

    const Atlas::IndexType GBZ_dim = 3;
    PolyTools::Laurant char_poly(GBZ_dim + 1);

    std::uniform_int_distribution<unsigned int> random_chart_label_int(0, 0x1<<GBZ_dim);

    /* set char poly */
    std::vector<int> orders = {
        0, 1, 0, 0,
        0, 1, -2, 0,
        1, 1, -1, -1,
        2, 0, 0, -2,
        0, 0, 0, -1,
        1, 0, 0, 0,
        2, -2, 0, 0,
        0, 0, -2, 0,
        0, -1, 0, 0,
        0, 0, 0, 1
    };
    PolyTools::ScalarVec coeffs = {
        1,2,3,4,5,6,7,8,9,10
    };
    char_poly.set_Laurant_by_terms(coeffs, orders);

    // /* set function */
    CharPolyFun<GBZ_dim> char_poly_fun(char_poly);

    // /* generate random points */
    // Atlas::GBZBaseManifold<GBZ_dim>::PointDataType GBZ_point;
    // for(auto it = GBZ_point.coords.begin(); it != GBZ_point.coords.end(); it++)
    // {
    //     (*it) = Atlas::ComplexType(normal_dist(gen), normal_dist(gen));
    // }
    // GBZ_point.chart_labels.fill(0);

    // /* evaluate */
    // std::cout << char_poly_fun.eval(GBZ_point) << '\n';

    // print_vector(GBZ_point.coords);
    // std::cout << '\n';
    // std::cout << char_poly_fun.char_poly_list[0]->num.to_str({"E", "x", "y", "z"}) << '\n';
    // print_vector(char_poly_fun.char_poly_list[0]->denom_orders);
    // std::cout << '\n';
    
    // Atlas::GBZBaseManifold<GBZ_dim> viewer(GBZ_point.coords.data(), GBZ_point.chart_labels.data());
    // for(unsigned int i = 0; i < 20; i++)
    // {
    //     unsigned int curr_chart_labels_int = random_chart_label_int(gen);
    //     Atlas::GBZBaseManifold<GBZ_dim>::ChartLabelType curr_chart_labels;
    //     binary_to_chart_labels<GBZ_dim>(curr_chart_labels, curr_chart_labels_int);
    //     viewer.to_chart(curr_chart_labels);
    //     print_vector(GBZ_point.chart_labels);
    //     std::cout << '\n';
    //     std::cout << char_poly_fun.eval(GBZ_point) <<'\n';
    // }

    return 0;

}

int main()
{
    // return test_conversion();
    // return test_polynomial_initialization();
    return test_evaluation();
}