'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-09-01
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import phcpy

def main():
    sol = phcpy.solver.solve([
        "2 * x + y + 1;",
        "x - y - 1;"
    ])
    print(sol)

if __name__ == '__main__':
    main()