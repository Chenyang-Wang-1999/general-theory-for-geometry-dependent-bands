/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-17
 * @lastModified  2023-07-20
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */


#include "atlas.hpp"
#include "GBZ_manifolds.hpp"
#include <array>
#include <vector>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

using namespace Atlas;
template<typename PointViewType>
class PythonPointDataArr: public std::vector<typename PointViewType::PointDataType>
{
    void get_coords(IndexType point_id,  typename PointViewType::CoordType & coord_list)
    {
        for(IndexType curr_dim = 0; curr_dim < PointViewType::dim; curr_dim ++)
        {
            coord_list[curr_dim] = std::vector<typename PointViewType::PointDataType>::operator[](point_id).coords[curr_dim];
        }
    }

    void get_chart_label(IndexType point_id,  typename PointViewType::ChartLabelType & chart_label)
    {
        for(ChartIdType curr_dim = 0; curr_dim < PointViewType::chart_id_dim; curr_dim ++)
        {
            chart_label[curr_dim] = std::vector<typename PointViewType::PointDataType>::operator[](point_id).chart_labels[curr_dim];
        }
    }

    void set_coords(IndexType point_id, const typename PointViewType::CoordType & coord_list)
    {
        for(IndexType curr_dim = 0; curr_dim < PointViewType::dim; curr_dim ++)
        {
            std::vector<typename PointViewType::PointDataType>::operator[](point_id).coords[curr_dim] = coord_list[curr_dim];
        }
    }

    void set_chart_label(IndexType point_id, const typename PointViewType::ChartLabelType & chart_label)
    {
        for(ChartIdType curr_dim = 0; curr_dim < PointViewType::chart_id_dim; curr_dim ++)
        {
            std::vector<typename PointViewType::PointDataType>::operator[](point_id).chart_labels[curr_dim] = chart_label[curr_dim];
        }
    }
};

template<class _PointViewType>
class PyPointView:public _PointViewType
{
    public:
        typedef _PointViewType PointViewType;
        typedef typename PointViewType::ScalarType ScalarType;
        typedef typename PointViewType::PointDataType PointDataType;
        typedef typename PointViewType::ChartLabelType ChartLabelType;
        typedef typename PointViewType::CoordType CoordType;
        const static ChartIdType chart_id_dim = PointViewType::chart_id_dim;
        const static IndexType dim = PointViewType::dim;
    
    PyPointView(){}
    
    void reinit_data_python(PointDataType& point_data)
    {
        reinit_data(point_data.coords.data(), point_data.chart_labels.data());
    }

    void reinit_data_to_vec_python(std::vector<PointDataType> & point_data_list, IndexType point_id)
    {
        reinit_data_python(point_data_list[point_id]);
    }
};

namespace py=pybind11;


typedef PyPointView<CP1> PyCP1;
typedef CP1::PointDataType PyCP1Data;

typedef PyPointView<ComplexN<1>> PyComplex1;
typedef ComplexN<1>::PointDataType PyComplex1Data;

typedef PyPointView<GBZBaseManifold<1>> PyGBZBase1;
typedef GBZBaseManifold<1>::PointDataType PyGBZBase1Data;

typedef PyPointView<GBZBaseManifold<2>> PyGBZBase2;
typedef GBZBaseManifold<2> PyGBZBase2Data;

typedef PyPointView<GBZBaseManifold<3>> PyGBZBase3;
typedef GBZBaseManifold<3> PyGBZBase3Data;

typedef PyPointView<GBZBaseManifold<4>> PyGBZBase4;
typedef GBZBaseManifold<4>::PointDataType PyGBZBase4Data;

typedef std::array<ComplexType, 1> ComplexArr1;
PYBIND11_MAKE_OPAQUE(ComplexArr1);
typedef std::array<ComplexType, 2> ComplexArr2;
PYBIND11_MAKE_OPAQUE(ComplexArr2);
typedef std::array<ComplexType, 3> ComplexArr3;
PYBIND11_MAKE_OPAQUE(ComplexArr3);
typedef std::array<ComplexType, 4> ComplexArr4;
PYBIND11_MAKE_OPAQUE(ComplexArr4);
typedef std::array<ComplexType, 5> ComplexArr5;
PYBIND11_MAKE_OPAQUE(ComplexArr5);
typedef std::array<ChartIdType, 1> ChartIdArr1;
PYBIND11_MAKE_OPAQUE(ChartIdArr1);
typedef std::array<ChartIdType, 2> ChartIdArr2;
PYBIND11_MAKE_OPAQUE(ChartIdArr2);
typedef std::array<ChartIdType, 3> ChartIdArr3;
PYBIND11_MAKE_OPAQUE(ChartIdArr3);
typedef std::array<ChartIdType, 4> ChartIdArr4;
PYBIND11_MAKE_OPAQUE(ChartIdArr4);


/* macros for type conversion */
#define DEFINE_STD_ARRAY(STDARRNAME, ScalarName) \
{ \
    py::class_<STDARRNAME>(m, #STDARRNAME) \
    .def(py::init<>()) \
    .def("__getitem__", [](const STDARRNAME & v, unsigned int n){return v[n];}) \
    .def("__setitem__", [](STDARRNAME & v, unsigned int n, ScalarName x){v[n] = x;}) \
    .def("__len__", &STDARRNAME::size); \
}

PYBIND11_MODULE(test_module, m)
{
    DEFINE_STD_ARRAY(ChartIdArr1, ChartIdType)
    DEFINE_STD_ARRAY(ChartIdArr2, ChartIdType)
    DEFINE_STD_ARRAY(ChartIdArr3, ChartIdType)
    DEFINE_STD_ARRAY(ChartIdArr4, ChartIdType)
    DEFINE_STD_ARRAY(ComplexArr1, ComplexType)
    DEFINE_STD_ARRAY(ComplexArr2, ComplexType)
    DEFINE_STD_ARRAY(ComplexArr3, ComplexType)
    DEFINE_STD_ARRAY(ComplexArr4, ComplexType)
    DEFINE_STD_ARRAY(ComplexArr5, ComplexType)
    py::class_<PyGBZBase4Data>(m, "PyGBZBase4Data")
        .def(py::init<>())
        .def_readwrite("coords", &PyGBZBase4Data::coords)
        .def_readwrite("chart_labels", &PyGBZBase4Data::chart_labels);
}
