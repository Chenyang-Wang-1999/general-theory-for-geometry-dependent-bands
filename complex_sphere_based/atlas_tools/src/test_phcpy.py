'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-10-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from phcpy import solver

def test_Laurent():
    eq_system = [
        "x*y + y^(-1)+1;",
        "x+y-x^2;"
    ]
    data = solver.solve(eq_system, dictionary_output=True)
    for sol in data:
        print(sol)
        x = sol['x']
        y = sol['y']
        print(x*y + 1/y + 1)
        print(x + y - x**2)

if __name__ == "__main__":
    test_Laurent()