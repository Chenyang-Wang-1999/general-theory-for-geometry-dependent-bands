/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2026-02-09
 * @lastModified  2026-02-09
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */


#include <complex>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

typedef std::complex<double> complex_t;

complex_t beta_series(
    int ell,
    double kappa,
    complex_t beta
)
{
    return exp(-kappa * ell) * (pow(beta, ell) + pow(beta, -ell));
}

complex_t dbeta_series(
    int ell,
    double kappa,
    complex_t beta
)
{
    return exp(-kappa * ell) * (pow(beta, ell - 1) * (ell * 1.0) - pow(beta, -ell - 1) * (ell * 1.0));
}

namespace py=pybind11;
PYBIND11_MODULE(long_range_cpp, m)
{
    m.def("beta_series", &beta_series);
    m.def("dbeta_series", &dbeta_series);
}
