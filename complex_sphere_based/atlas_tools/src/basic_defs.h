/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2022-12-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved
 */

#ifndef BASIC_DEFS_H
#define BASIC_DEFS_H


#include <complex>
#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>

namespace Atlas{
    typedef unsigned short ChartIdType;
    typedef unsigned int IndexType;
    typedef double RealType;
    typedef std::complex<RealType> ComplexType;
}

namespace PolyTools{

#define SCALAR_MODE 0 
typedef Atlas::ComplexType Scalar;
typedef Atlas::ComplexType VarScalar;
typedef Atlas::IndexType IndexType;


typedef std::vector<Scalar> ScalarVec;
typedef std::vector<VarScalar> VarScalarVec;
typedef std::vector<IndexType> IndexVec; 
typedef enum{GT, EQ, LT, GEQ, LEQ} CompareResult;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> EigenMatrixX;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> EigenVectorX;
typedef Eigen::Matrix<VarScalar, Eigen::Dynamic, Eigen::Dynamic> EigenVarMatrixX;
typedef Eigen::Matrix<VarScalar, Eigen::Dynamic, 1> EigenVarVectorX;
#define STDOUT std::cout
#define STDERR std::cerr
#define EPS 1e-16
#define EXP_BASE 1
#define ABS_FUN abs
}// namespace PolyTools

#endif