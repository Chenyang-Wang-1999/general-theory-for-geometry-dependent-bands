'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-24
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from .util import *
from phcpy import solver
from scipy import io as sio
import poly_tools as pt
import GBZ_manifold as Gm

class GBZContinuityChecker(CriticalChecker):
    tol: float
    using_raw: bool
    def __init__(self, tol):
        self.tol = tol
        self.using_raw = False
    
    def __call__(self, param, old_vec, new_vec) -> bool:
        ''' check whether all the distance is less than tol 
                if not continuous, return "True" 
        '''
        if(len(old_vec) >= len(new_vec)):
            smaller_vec = new_vec
            larger_vec = old_vec
        else:
            smaller_vec = old_vec
            larger_vec = new_vec
        
        for point in smaller_vec:
            dist, _ = Gm.calculate_min_distance(point, larger_vec)
            if(dist > self.tol):
                return True
        else:
            return False

class aGBZEquation1D(ParamFun):
    char_poly:pt.CLaurent # E, beta
    grad_char_poly: list[pt.CLaurent]
    def __init__(self, char_poly:pt.CLaurent) -> None:
        self.char_poly = char_poly
        self.grad_char_poly = [char_poly.derivative(j) for j in range(2)]
        self.df_beta = [self.grad_char_poly[1].derivative(j) for j in range(2)]

    def __call__(self, x):
        exp_iphi = np.exp(1j * self.params[0])
        x_arr = pt.CVarScalarVec(x)
        f = np.zeros((2,), dtype=complex)
        f[0] = self.char_poly.eval(x_arr)
        pd_f = np.zeros((2,2), dtype=complex)
        for j in range(2):
            pd_f[0,j] = self.grad_char_poly[j].eval(x_arr)

        if(np.abs(exp_iphi-1) < 1e-10):
            f[1] = self.grad_char_poly[1].eval(x_arr)
            for j in range(2):
                pd_f[1,j] = self.df_beta[j].eval(x_arr)
        else:

            x_phi = x.copy()
            x_phi[1] *= exp_iphi
            x_phi_arr = pt.CVarScalarVec(x_phi)
            f[1] = self.char_poly.eval(x_phi_arr)
            for j in range(2):
                pd_f[1,j] = self.grad_char_poly[j].eval(x_phi_arr)
            pd_f[1,1] *= exp_iphi

        return f, pd_f 

    def get_equations(self):
        exp_iphi = np.exp(1j * self.params[0])
        if(np.abs(exp_iphi-1) < 1e-10):
            return [self.char_poly.to_str(pt.CStrVec(['x','y'])), self.grad_char_poly[1].to_str(pt.CStrVec(['x','y']))]
        else:
            return [self.char_poly.to_str(pt.CStrVec(['x','y'])), self.char_poly.scale_var(pt.CScalarVec([1, exp_iphi])).to_str(pt.CStrVec(['x','y']))]

class aGBZEquation1DGBZCP1(ParamFun):
    aGBZ_equation_list: list[aGBZEquation1D]
    def __init__(self, char_poly: pt.CLaurent) -> None:
        self.aGBZ_equation_list = [
            aGBZEquation1D(char_poly),
            aGBZEquation1D(char_poly.flip_variable(1))
        ]
    
    def set_params(self, params):
        self.aGBZ_equation_list[0].set_params(params.copy())
        params[0] = -params[0]
        self.aGBZ_equation_list[1].set_params(params.copy())
    
    def __call__(self, x: Gm.GBZPointData):
        return self.aGBZ_equation_list[x.chart_labels[0]](x.coords)
    
    def get_callable(self, chart_labels: list):
        return self.aGBZ_equation_list[chart_labels[0]]
    
class aGBZEquation1DCP1(ParamFun):
    aGBZ_equation_list: list[aGBZEquation1D]
    def __init__(self, char_poly: pt.CLaurent) -> None:
        self.aGBZ_equation_list = [
            aGBZEquation1D(char_poly),
            aGBZEquation1D(char_poly.flip_variable(1)),
            aGBZEquation1D(char_poly.flip_variable(0)),
            aGBZEquation1D(char_poly.flip_variable(0).flip_variable(1))
        ]
    
    def set_params(self, params):
        self.aGBZ_equation_list[0].set_params(params.copy())
        self.aGBZ_equation_list[2].set_params(params.copy())
        params[0] = -params[0]
        self.aGBZ_equation_list[1].set_params(params.copy())
        self.aGBZ_equation_list[3].set_params(params.copy())
    
    def __call__(self, x: Gm.GBZPointData):
        return self.aGBZ_equation_list[x.chart_labels[0] * 2 + x.chart_labels[1]](x.coords)
    
    def get_callable(self, chart_labels: list):
        return self.aGBZ_equation_list[chart_labels[0] * 2 + chart_labels[1]]

def phcpy_full_solve(GBZ_equation:aGBZEquation1D):
    eq_str = GBZ_equation.get_equations()
    # all_sols = solver.solve(eq_str, dictionary_output=True, precision='qd')
    all_sols = solver.solve(eq_str, dictionary_output=True)
    point_list = []
    for sol in all_sols:
        curr_point = np.array([complex(sol['x']), complex(sol['y'])])
        res = complex_root(GBZ_equation, curr_point, jac=True)
        for _ in range(sol['m']):
            point_list.append(res.x)
    point_list = np.array(point_list)
    return point_list

def phcpy_GBZCP1_full_solve(GBZ_equation: aGBZEquation1DGBZCP1):
    point_list = phcpy_full_solve(GBZ_equation.get_callable([0]))
    GBZ_point_vec = Gm.GBZPointVec()
    for j in range(point_list.shape[0]):
        GBZ_point_vec.append(
            Gm.GBZPointData(point_list[j,:], [0])
        )
    Gm.batch_to_standard(GBZ_point_vec)

    root_fun = BatchCP1Root(Gm.GBZPointData, jac=True)    
    GBZ_point_vec = root_fun(GBZ_equation, GBZ_point_vec)

    return GBZ_point_vec
    

def phcpy_CP1_full_solve(GBZ_equation: aGBZEquation1DCP1):
    point_list = phcpy_full_solve(GBZ_equation.get_callable([0,0]))
    GBZ_point_vec = Gm.CP1PointVec()
    for j in range(point_list.shape[0]):
        GBZ_point_vec.append(
            Gm.CP1PointData(point_list[j,:], [0,0])
        )
    Gm.batch_to_standard(GBZ_point_vec)

    root_fun = BatchCP1Root(Gm.CP1PointData, jac=True)    
    GBZ_point_vec = root_fun(GBZ_equation, GBZ_point_vec)

    return GBZ_point_vec

class CP1DistanceChecker:
    tol: float
    def __init__(self, tol: float):
        self.tol = tol
    
    def __call__(self, curr_sol: Gm.GBZPointVec):
        dis = Gm.check_min_distance(curr_sol, self.tol)
        if(dis > self.tol):
            return True
        else:
            return False

class BatchCP1Root:
    root_options: dict
    def __init__(self, dtype, **root_options) -> None:
        self.dtype = dtype
        if(dtype == Gm.GBZPointData):
            self.vec_type = Gm.GBZPointVec
        elif(dtype == Gm.CP1PointData):
            self.vec_type = Gm.CP1PointVec
        self.root_options = root_options
    
    def __call__(self, fun, all_x0) -> Gm.GBZPointVec:
        all_sols = self.vec_type()
        for point in all_x0:
            curr_fun = fun.get_callable(point.chart_labels)
            res = complex_root(curr_fun, np.asarray(point.coords), **self.root_options)
            if(res.success):
                is_successful = True
            else:
                curr_x = res.x
                f_val, f_jac = curr_fun(curr_x)
                if(np.linalg.norm(f_val) < 1e-12):
                    is_successful = True
                else:
                    is_successful = False

            if(is_successful):
                all_sols.append(self.dtype(
                    res.x, point.chart_labels
                ))
            else:
                return None
        Gm.batch_to_standard(all_sols)
        return all_sols
    
    def solve_one(self, fun, point):
        curr_fun = fun.get_callable(point.chart_labels)
        res = complex_root(curr_fun, np.asarray(point.coords), **self.root_options)
        if(res.success):
            is_successful = True
        else:
            curr_x = res.x
            f_val, f_jac = curr_fun(curr_x)
            if(np.linalg.norm(f_val) < 1e-12):
                is_successful = True
            else:
                is_successful = False

        if(is_successful):
            return self.dtype(
                res.x, point.chart_labels
            )
        else:
            raise_warning("Warning(BatchCP1Root.solve_one): continuous solver fails. Using full solver.")
            if(self.dtype == Gm.CP1PointData):
                all_sols = phcpy_CP1_full_solve(fun)
            else:
                all_sols = phcpy_GBZCP1_full_solve(fun)
            _, min_id = Gm.calculate_min_distance(point, all_sols)
            return all_sols[min_id]


# def solve_aGBZ(char_poly:pt.CLaurent, N_points = 200):
#     ''' solve GBZ for 1D system 
#         char_poly: characteristic polynomial of the Hamiltonian, f(E, beta)
#     '''
#     aGBZ_eq = aGBZEquation1D(char_poly)
#     distance_checker = PointSetDistanceChecker(1e-2, 50)
#     critical_checker = ContinuityCriticalChecker(0.5)
#     root_solver = BatchComplexRoot(jac=True)
#     solver_control = BatchSolverController(phcpy_full_solve, root_solver, aGBZ_eq, distance_checker, critical_checker)

#     exp_phi_loop = np.exp(1j*np.linspace(np.pi/N_points, np.pi, N_points)).reshape((-1,1))

#     solver_control.set_path(exp_phi_loop)
#     solver_control.evolve_to_dest()

#     exp_phi_loop = solver_control.param_path
#     return (np.asarray(solver_control.value_list), exp_phi_loop)
    
def match_CP1_points(curr_sol, next_sol):
    if(isinstance(curr_sol, Gm.GBZPointVec)):
        result = Gm.GBZPointVec()
    elif(isinstance(curr_sol, Gm.CP1PointVec)):
        result = Gm.CP1PointVec()
    else:
        raise ValueError("unsupported type")
    Gm.match_point_list(curr_sol, next_sol, result)
    return result

   