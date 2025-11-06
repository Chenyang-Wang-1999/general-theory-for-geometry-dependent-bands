'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-20
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import c_solver_2D as csolver
import numpy as np
import matplotlib.pyplot as plt

def test_char_poly():
    # vx1 = 1.0 + np.random.randn() * 0.5
    # vx2 = 1.0 + np.random.randn() * 0.5
    # ux1 = 3.0 + np.random.randn() * 0.5
    # ux2 = 3.0 + np.random.randn() * 0.5
    # uy1 = 4.0 + np.random.randn() * 0.5
    # uy2 = 4.0 + np.random.randn() * 0.5
    vx1 = 1.0
    vx2 = 1.0
    ux1 = 3.0
    ux2 = 3.0
    uy1 = 4.0
    uy2 = 4.0
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    return char_poly

def test_equation_consistency():
    char_poly = test_char_poly()
    solver_2d = csolver.GBZConditionChecker2D(np.array([1,1.0,np.exp(np.pi/4*1j)]), np.exp(1j*np.pi/4),
                                              char_poly, 1)
    print(solver_2d.char_poly.num.to_str(csolver.CStrVec(["x", "y", "z"])))
    solver_2d.set_effective_poly(1.0)
    eq_str = solver_2d.full_solve_str()
    print(eq_str)

    x0 = np.random.randn(2) + 1j * np.random.randn(2)
    val, jac = solver_2d.continuation_solver_fun(x0)

    # print(val)
    # x = x0[0]
    # y = x0[1]
    # i = 1j
    # print(+(4+0*i)+(-8+0*i)*(y**2)+(4+0*i)*(y**4)+(-5+0*i)*x*y+(-5+0*i)*x*(y**3)+(1+0*i)*(x**2)*(y**2) - val[0])
    # print(+(4+0*i)+(-4.89858719658941283e-16-8*i)*(y**2)+(-4+4.89858719658941283e-16*i)*(y**4)+(-3.53553390593273775-3.53553390593273731*i)*x*y+(3.53553390593273731-3.53553390593273775*i)*x*(y**3)+(6.12323399573676604e-17+1*i)*(x**2)*(y**2)-val[1])

    # test derivative
    dx0 = (np.random.randn(2) + 1j * np.random.randn(2))*1e-4
    val1, jac = solver_2d.continuation_solver_fun(x0 + dx0)
    print(val1-val)
    print(jac@dx0)
    print(val1-val - jac@dx0)


'''
    In the following code, we will use 2D HN model to test the winding solver
'''

def HN_2d_char_poly(Jx1, Jx2, Jy1, Jy2):
    coeffs = csolver.CScalarVec([
        1, -Jx1, -Jx2, -Jy1, -Jy2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        1, 0, 0,
        0, -1, 1,
        0, 1, -1,
        0, 0, -1,
        0, 0, 1
    ])
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    return char_poly

def HN_2d_solution_x_major(Jx1, Jx2, Jy1, Jy2, expiphi):
    expik = np.sqrt(expiphi)
    cosk = expik.real
    beta1 = np.sqrt(Jx1 * Jy1/(Jx2 * Jy2)) / expik[0]
    beta2 = (1.0/expik[1]) * np.sqrt((Jx2*beta1 + Jy1) / (Jx1/beta1 + Jy2))
    E = 2 * cosk[1]*np.sqrt(2*cosk[0]*np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) + Jx1 * Jx2 + Jy1 * Jy2)
    return (E, beta1, beta2)

def beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, beta, expiphi2):
    return (Jx2 * beta + Jy1)/(Jx1/beta + Jy2)/expiphi2

def main_test_HN():
    Jx1 = 1
    Jx2 = 1
    Jy1 = 0.8j
    Jy2 = 0.8j
    expiphi = np.exp(1j*np.array([np.pi/3, np.pi/2]))
    E, beta1, beta2 = HN_2d_solution_x_major(Jx1, Jx2, Jy1, Jy2, expiphi)
    char_poly = HN_2d_char_poly(Jx1, Jx2, Jy1, Jy2)
    print(char_poly.eval(csolver.CScalarVec([E, beta1, beta2])))

    winding_solver = csolver.GBZConditionChecker2D(np.array([E, beta1, beta2]), expiphi[1], 
                                char_poly, 1)
    beta_loop = beta1 * np.exp(1j*np.linspace(0, 2*np.pi, 100))

    char_poly_vals, sol_points  = winding_solver.eval_param_path(
        beta_loop.reshape((-1,1)), np.array([E, beta2]),
        rel_r_detour=0.01,
        N_points_loop=100, N_points_detour=50
    )

    beta2_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, beta_loop, expiphi[1])
    plt.plot((sol_points[:,1]**2).real, (sol_points[:,1]**2).imag)
    plt.plot(beta2_theo.real, beta2_theo.imag, '--')
    plt.show()
    # sol_check = beta2_theo**2 - sol_points[:,1]**2
    # E_check = 2 * np.sqrt(expiphi[1]).real * np.sqrt(Jx2 * Jy2 * beta_loop + Jx1 * Jy1/beta_loop + Jx1 * Jx2 + Jy1 * Jy2)
    # plt.plot(sol_check.real, sol_check.imag)
    # plt.show()

    # plt.plot(sol_points[:,0].real, sol_points[:,0].imag)
    # plt.plot(E_check.real, E_check.imag, '--')
    # plt.show()

    # print(char_poly_vals.shape)
    # plt.plot(char_poly_vals.real, char_poly_vals.imag)
    # plt.plot(char_poly_vals[0].real, char_poly_vals[0].imag, 'rx')
    # plt.show()
    print(beta2)
    
    points_circ_1, points_circ_2, points_inner_circ_1, points_inner_circ_2, points_path_12, points_path_21, circ_1, circ_2, inner_circ_1, inner_circ_2, path_12, path_21 = csolver.check_GBZ_condition(
        char_poly, np.array([E, beta1, beta2]), expiphi, 1
    )
    # plt.plot((points_circ_1[:,1]**2).real, (points_circ_1[:,1]**2).imag)
    # points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, circ_1, expiphi[1])
    # plt.plot(points_theo.real, points_theo.imag, '--')
    # plt.show()

    # plt.plot((points_circ_2[:,1]**2).real, (points_circ_2[:,1]**2).imag)
    # points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, circ_2, expiphi[1])
    # plt.plot(points_theo.real, points_theo.imag, '--')
    # plt.show()

    # plt.plot((points_inner_circ_1[:,1]**2).real, (points_inner_circ_1[:,1]**2).imag)
    # points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, inner_circ_1, expiphi[1])
    # plt.plot(points_theo.real, points_theo.imag, '--')
    # plt.show()

    # plt.plot((points_inner_circ_2[:,1]**2).real, (points_inner_circ_2[:,1]**2).imag)
    # points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, inner_circ_1, expiphi[1])
    # plt.plot(points_theo.real, points_theo.imag, '--')
    # plt.show()

    plt.plot((points_path_12[:,1]**2).real, (points_path_12[:,1]**2).imag)
    points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, path_12, expiphi[1])
    plt.plot(points_theo.real, points_theo.imag, '--')
    plt.plot((beta2**2).real, (beta2**2).imag, 'x')
    plt.show()

    # plt.plot((points_path_21[:,1]**2).real, (points_path_21[:,1]**2).imag)
    # points_theo = beta2sq_x_major(Jx1, Jx2, Jy1, Jy2, path_21, expiphi[1])
    # plt.plot(points_theo.real, points_theo.imag, '--')
    # plt.show()



if __name__ == '__main__':
    # test_equation_consistency()
    main_test_HN()