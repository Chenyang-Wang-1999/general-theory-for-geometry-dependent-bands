'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-10-01
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import c_solver_2D as csolver

def test_Groebner():
    f = csolver.CPolyLinkedList(2)
    g = csolver.CPolyLinkedList(2)
    h = csolver.CPolyLinkedList(2)

    coeffs = csolver.CScalarVec([1, -1])
    orders = csolver.CIndexVec([3, 2, 0, 1])
    f.batch_add_elements(coeffs, orders)

    coeffs = csolver.CScalarVec([1, 1, -9])
    orders = csolver.CIndexVec([2, 0, 0, 2, 0,0])
    g.batch_add_elements(coeffs, orders)

    print(f.to_str(["x", "y"]))
    print(g.to_str(["x", "y"]))
    poly_basis = csolver.CPolyDeque()
    poly_basis.push_back(f)
    poly_basis.push_back(g)

    csolver.buchberger(poly_basis, 1e-12)
    for j in range(poly_basis.size()):
        curr_poly = csolver.CPolyLinkedList(2)
        poly_basis.get_poly(j, curr_poly)
        print(curr_poly.to_str(csolver.CStrVec(["x", "y"])))

if __name__ == "__main__":
    test_Groebner()