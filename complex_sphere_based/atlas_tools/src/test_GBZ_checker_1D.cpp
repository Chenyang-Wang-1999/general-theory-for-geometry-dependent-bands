/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-09-01
 * @lastModified  2023-09-06
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include <atlas/atlas.hpp>
#include <atlas/GBZ_manifolds.hpp>
#include <aGBZ_2d.hpp>

int main_test_winding()
{
    Atlas::RealType 
        vx1 = 1.0,  
        vx2 = 1.0,
        ux1 = 3.0,
        ux2 = 3.0,
        uy1 = 4.0,
        uy2 = 4.0;
    PolyTools::ScalarVec coeffs = {
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    };

    std::vector<int> orders = {
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    };

    // set char poly
    PolyTools::Laurant char_poly(3);
    char_poly.set_Laurant_by_terms(coeffs, orders);

    // set checking object
    GBZCondition1DChecker checker(
        char_poly, 1
    );

    Atlas::ComplexType 
        E_val = Atlas::ComplexType(0.11263782, 2.52569303e-10),
        major_beta = Atlas::ComplexType(0.798948993178733, -0.6013987954043352),
        beta = Atlas::ComplexType(0.99699551, -7.74593996e-02);
    

    bool check_result = checker.GBZ_condition_check(E_val, major_beta, beta, 1e-2*M_PI, 200, 50, 1e-3);
    
    STDOUT << check_result << '\n';
    return 0;
}

int main(){
    return main_test_winding();
}