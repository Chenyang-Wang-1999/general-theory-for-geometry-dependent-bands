/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-20
 * @lastModified  2023-07-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#include "src/atlas/atlas.hpp"
#include "src/atlas/GBZ_manifolds.hpp"
#include <iostream>
using namespace Atlas;

int main()
{
    std::cout << GBZBaseManifold<1>::dim << '\n';
    return 0;
}