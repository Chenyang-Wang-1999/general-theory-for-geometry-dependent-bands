/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2024-03-18
 * @lastModified  2025-03-31
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

/* Python interface for GBZ manifold */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <../src/atlas/GBZ_manifolds.hpp>
#include <tuple>

namespace py=pybind11;
using namespace std;

/* define point lists */
typedef Atlas::GBZBaseManifold<1>::PointDataType GBZPointData;
typedef std::vector<GBZPointData> GBZPointVec;

typedef Atlas::CP1Product<2>::PointDataType CP1PointData;
typedef std::vector<CP1PointData> CP1PointVec;


template<typename PointViewType> 
py::tuple serialize_point_data(
    const typename PointViewType::PointDataType & point
)
{
    return py::make_tuple(point.coords, point.chart_labels);
}

template<typename PointViewType> 
typename PointViewType::PointDataType deserialize_point_data(
    const std::tuple<typename PointViewType::CoordType, typename PointViewType::ChartLabelType> & state
)
{
    typename PointViewType::PointDataType curr_point;
    // curr_point.coords = state[0].cast<typename PointViewType::CoordType>();
    // curr_point.chart_labels = state[1].cast<typename PointViewType::ChartLabelType>();
    std::tie(curr_point.coords, curr_point.chart_labels) = state;
    return curr_point;
}

template<typename PointViewType> 
py::tuple serialize_point_vector(
    const std::vector<typename PointViewType::PointDataType> & point_vec
)
{
    py::list state;
    for(const auto & point : point_vec)
    {
        state.append(serialize_point_data<PointViewType>(point));
    }
    return py::tuple(state);
}


template<typename PointViewType> 
std::vector<typename PointViewType::PointDataType> deserialize_point_vector(
    const py::tuple & state
)
{
    std::vector<typename PointViewType::PointDataType> point_vector;
    for(const auto & curr_state : state)
    {
        point_vector.push_back(deserialize_point_data<PointViewType>(
            curr_state.cast<std::tuple<
                typename PointViewType::CoordType, 
                typename PointViewType::ChartLabelType>>()));
    }
    return point_vector;
}

typedef std::array<double, 3> Point3D;
typedef std::vector<Point3D> Point3DVec;

/* define 3D coordinate retrival */
Point3D get_point_3D_coord(GBZPointData & GBZ_point_data)
{
    Atlas::GBZBaseManifold<1> viewer;
    viewer.reinit_data(GBZ_point_data.coords.data(), GBZ_point_data.chart_labels.data());
    Atlas::CP1 cp1_viewer = viewer.get_right().get(0);
    Point3D real_coord;
    cp1_viewer.embed_into_affine(real_coord);
    return real_coord;
}

void batch_get_point_3D_coord(GBZPointVec & GBZ_point_vec, Point3DVec & real_coords)
{
    real_coords.clear();
    for(auto point:GBZ_point_vec)
    {
        real_coords.push_back(get_point_3D_coord(point));
    }
}


template<typename PointViewType>
py::tuple py_calculate_min_distance(typename PointViewType::PointDataType &point,
    std::vector<typename PointViewType::PointDataType> &point_vec)
{
    Atlas::IndexType min_id;
    Atlas::RealType min_dist = Atlas::calculate_min_distance<PointViewType>(point, point_vec, min_id);
    return py::make_tuple(min_dist, min_id);
}

Atlas::RealType get_CP1Product_interp(
    Atlas::CP1Product<2>::PointDataType & point1, 
    Atlas::CP1Product<2>::PointDataType & point2,
    Atlas::RealType norm_cos_theta,
    Atlas::RealType s_ini=0.5,
    Atlas::RealType tol=1e-12,
    Atlas::IndexType max_iter=10000 
)
{
    Atlas::CP1Product<2> 
        viewer1(point1.coords.data(), point1.chart_labels.data()), 
        viewer2(point2.coords.data(), point2.chart_labels.data());
    return Atlas::interp_CP1_points_given_norm(
        viewer1.get(0),
        viewer2.get(0),
        norm_cos_theta,
        s_ini,
        tol,
        max_iter
    );
}

template<Atlas::IndexType N>
Atlas::RealType get_CP1_distance_CP1(
    typename Atlas::CP1Product<N>::PointDataType & point1,
    typename Atlas::CP1Product<N>::PointDataType & point2,
    int which_beta
){
    Atlas::CP1Product<N> viewer1(point1.coords.data(), point1.chart_labels.data());
    Atlas::CP1Product<N> viewer2(point2.coords.data(), point2.chart_labels.data());
    Atlas::CP1 cp1_1, cp1_2;
    cp1_1 = viewer1.get(which_beta);
    cp1_2 = viewer2.get(which_beta);
    return Atlas::distance(cp1_1, cp1_2);
}

template<Atlas::IndexType N>
Atlas::RealType get_CP1_distance_GBZ(
    typename Atlas::GBZBaseManifold<N>::PointDataType & point1,
    typename Atlas::GBZBaseManifold<N>::PointDataType & point2,
    int which_beta
){
    Atlas::GBZBaseManifold<N> viewer1(point1.coords.data(), point1.chart_labels.data());
    Atlas::GBZBaseManifold<N> viewer2(point2.coords.data(), point2.chart_labels.data());
    Atlas::CP1 cp1_1, cp1_2;
    cp1_1 = viewer1.get_right().get(which_beta);
    cp1_2 = viewer2.get_right().get(which_beta);
    return Atlas::distance(cp1_1, cp1_2);
}

template<typename PointViewType> void point_to_chart(
    typename PointViewType::PointDataType & point, 
    typename PointViewType::ChartLabelType & chart_labels
)
{
    PointViewType viewer(point.coords.data(), point.chart_labels.data());
    viewer.to_chart(chart_labels);
}


template<typename PointViewType>
bool check_duplicates(
    typename PointViewType::PointDataType & point,
    std::vector<typename PointViewType::PointDataType> & point_vec,
    Atlas::RealType glue_tol
)
{
    typedef Eigen::Matrix<Atlas::RealType, PointViewType::affine_dim, 1> VectorType;
    PointViewType viewer;
    array<Atlas::RealType, PointViewType::affine_dim> point_affine;
    viewer.reinit_data(point.coords.data(), point.chart_labels.data());
    viewer.embed_into_affine(point_affine);
    VectorType point_affine_eigen(point_affine.data());

    Atlas::IndexType min_idx;
    Atlas::RealType min_dist = Atlas::calculate_min_distance<PointViewType>(point, point_vec, min_idx);
    if(min_idx == 0 || min_idx == point_vec.size() - 1)
    {
        array<Atlas::RealType, PointViewType::affine_dim> end_point, next_point;
        if(min_idx == 0)
        {
            viewer.reinit_data(point_vec[0].coords.data(), point_vec[0].chart_labels.data());
            viewer.embed_into_affine(end_point);
            viewer.reinit_data(point_vec[1].coords.data(), point_vec[1].chart_labels.data());
            viewer.embed_into_affine(next_point);
        }
        else
        {
            viewer.reinit_data(
                point_vec[point_vec.size() - 1].coords.data(), 
                point_vec[point_vec.size() - 1].chart_labels.data());
            viewer.embed_into_affine(end_point);
            viewer.reinit_data(
                point_vec[point_vec.size() - 2].coords.data(), 
                point_vec[point_vec.size() - 2].chart_labels.data());
            viewer.embed_into_affine(next_point);
        }
        Eigen::Map<VectorType> 
            end_point_vec(end_point.data()), next_point_vec(next_point.data());
        VectorType  diff_vec = next_point_vec - end_point_vec,
                    curr_vec = point_affine_eigen - end_point_vec;
        Atlas::RealType cos_theta = curr_vec.dot(diff_vec) / (diff_vec.dot(diff_vec));
        if(cos_theta < 0 || cos_theta > 1)
        {
            return false;
        }
        else
        {
            Atlas::RealType min_dist = (diff_vec - curr_vec * cos_theta).norm();
            if(min_dist < glue_tol)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

    }
    else
    {
        if(min_dist < glue_tol)
        {
            return true;
        }
        else
        {
            array<Atlas::RealType, PointViewType::affine_dim> 
                prev_point, mid_point, next_point;
            viewer.reinit_data(
                point_vec[min_idx - 1].coords.data(),
                point_vec[min_idx - 1].chart_labels.data());
            viewer.embed_into_affine(prev_point);
            viewer.reinit_data(
                point_vec[min_idx].coords.data(),
                point_vec[min_idx].chart_labels.data());
            viewer.embed_into_affine(mid_point);
            viewer.reinit_data(
                point_vec[min_idx + 1].coords.data(),
                point_vec[min_idx + 1].chart_labels.data()
            ); 
            viewer.embed_into_affine(next_point);

            Eigen::Map<VectorType>
                prev_point_vec(prev_point.data()), 
                mid_point_vec(mid_point.data()),
                next_point_vec(next_point.data());
            VectorType diff_vec_1 = prev_point_vec - mid_point_vec,
                        diff_vec_2 = next_point_vec - mid_point_vec,
                        curr_vec = point_affine_eigen - mid_point_vec;
            Atlas::RealType cos_theta_1 = curr_vec.dot(diff_vec_1) / (diff_vec_1.dot(diff_vec_1)),
                            cos_theta_2 = curr_vec.dot(diff_vec_2) / (diff_vec_2.dot(diff_vec_2));
            if(cos_theta_1 < 0 || cos_theta_1 > 1)
            {
                if(cos_theta_2 < 0 || cos_theta_2 > 1)
                {
                    return false;
                }
                else
                {
                    Atlas::RealType min_dist = (diff_vec_2 - curr_vec * cos_theta_2).norm();
                    if(min_dist < glue_tol)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            else
            {
                Atlas::RealType min_dist = (diff_vec_1 - curr_vec * cos_theta_1).norm();
                if(min_dist < glue_tol)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
    }
}


template<typename PointViewType>
py::tuple remove_duplicated_points(
    vector<typename PointViewType::PointDataType> point_vec1,
    vector<typename PointViewType::PointDataType> point_vec2,
    // vector<tuple<Atlas::IndexType, Atlas::IndexType>> & point_map,
    Atlas::RealType glue_tol
)
{
    /* Find duplicated points in the point list. 
        If there is, cut point_vec1 into several parts.
        Each part contains a set of points that are not duplicated with point_vec2.
        Input:
            point_vec1: a list of points to be checked
            point_vec2: another list of points 
            glue_tol: tolerance for checking if two points are duplicated
        Output:
            point_map: a list of tuples, each tuple contains 
                the start index and end index (not included) of the sliced point_vec1
            return: true if there is duplicated points, false otherwise
    */
    vector<tuple<Atlas::IndexType, Atlas::IndexType>> point_map;
    bool has_duplicates = false;
    Atlas::IndexType start = 0;

    for(Atlas::IndexType i = 0; i < point_vec1.size(); ++i) {
        bool is_duplicated = check_duplicates<PointViewType>(
            point_vec1[i], point_vec2, glue_tol);

        if(is_duplicated) {
            has_duplicates = true;
            // Add non-duplicate segment
            if(i > start) {
                point_map.emplace_back(start, i);
            }
            start = i + 1;
        }
    }

    // Add the last segment if there are remaining points
    if(start < point_vec1.size()) {
        point_map.emplace_back(start, point_vec1.size());
    }

    // return has_duplicates;
    return py::make_tuple(has_duplicates, point_map);
}


PYBIND11_MAKE_OPAQUE(GBZPointVec);
PYBIND11_MAKE_OPAQUE(Point3DVec);
PYBIND11_MAKE_OPAQUE(CP1PointVec);

PYBIND11_MODULE(lib_GBZ_manifold, m)
{
    py::bind_vector<GBZPointVec>(m, "cpp_GBZPointVec")
        .def("copy", [](const GBZPointVec & v) { return GBZPointVec(v); })
        .def(py::pickle(
            &serialize_point_vector<Atlas::GBZBaseManifold<1>>, 
            &deserialize_point_vector<Atlas::GBZBaseManifold<1>>));
    py::bind_vector<CP1PointVec>(m, "cpp_CP1PointVec")
        .def("copy", [](const CP1PointVec & v) { return CP1PointVec(v); })
        .def(py::pickle(
            &serialize_point_vector<Atlas::CP1Product<2>>, 
            &deserialize_point_vector<Atlas::CP1Product<2>>));
    py::bind_vector<Point3DVec>(m, "cpp_GBZPoint3DVec");
    py::class_<GBZPointData>(m, "cpp_GBZPointData")
        .def(py::init<>())
        .def_readwrite("chart_labels", &GBZPointData::chart_labels)
        .def_readwrite("coords", &GBZPointData::coords);
    py::class_<CP1PointData>(m, "cpp_CP1PointData")
        .def(py::init<>())
        .def_readwrite("chart_labels", &CP1PointData::chart_labels)
        .def_readwrite("coords", &CP1PointData::coords);
    m.def("cpp_get_point_3D_coord", &get_point_3D_coord);
    m.def("cpp_batch_get_point_3D_coord", &batch_get_point_3D_coord);
    m.def("cpp_check_min_distance_GBZ", 
        py::overload_cast<GBZPointVec&, Atlas::RealType>(
            &Atlas::check_min_distance<Atlas::GBZBaseManifold<1>>));
    m.def("cpp_batch_to_standard_GBZ",
        &Atlas::batch_to_standard<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_match_point_list_GBZ", 
        &Atlas::match_point_list_equal_length<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_get_distance_GBZ",
        &Atlas::get_distance<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_check_min_distance_CP1", 
        py::overload_cast<CP1PointVec&, Atlas::RealType>(
            &Atlas::check_min_distance<Atlas::CP1Product<2>>));
    m.def("cpp_batch_to_standard_CP1",
        &Atlas::batch_to_standard<Atlas::CP1Product<2>>);
    m.def("cpp_match_point_list_CP1", 
        &Atlas::match_point_list_equal_length<Atlas::CP1Product<2>>);
    m.def("cpp_get_distance_CP1",
        &Atlas::get_distance<Atlas::CP1Product<2>>);
    m.def("cpp_calculate_min_distance_GBZ",
        &py_calculate_min_distance<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_calculate_min_distance_CP1",
        &py_calculate_min_distance<Atlas::CP1Product<2>>);
    m.def("cpp_get_CP1Product_interp",
        &get_CP1Product_interp);
    m.def("cpp_linear_interp_CP1", 
        &Atlas::linear_interp<Atlas::CP1Product<2>>);
    m.def("cpp_linear_interp_GBZ", 
        &Atlas::linear_interp<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_get_z_coord_GBZ", [](GBZPointData & point, int j){
        Atlas::GBZBaseManifold<1> point_view(point.coords.data(), point.chart_labels.data());
        return point_view.get_right().get(j).get_z_coord();
    });
    m.def("cpp_get_z_coord_CP1", [](CP1PointData & point, int j){
        Atlas::CP1Product<2> point_view(point.coords.data(), point.chart_labels.data());
        return point_view.get(j).get_z_coord();
    });
    m.def("cpp_get_CP1_distance_GBZ", &get_CP1_distance_GBZ<1>);
    m.def("cpp_get_CP1_distance_CP1", &get_CP1_distance_CP1<2>);
    m.def("cpp_to_chart_CP1", &point_to_chart<Atlas::CP1Product<2>>);
    m.def("cpp_to_chart_GBZ", &point_to_chart<Atlas::GBZBaseManifold<1>>);
    m.def("cpp_remove_duplicated_points_GBZ", &remove_duplicated_points<Atlas::GBZBaseManifold<1>>, "Remove duplicated points in GBZ point list.");
    m.def("cpp_remove_duplicated_points_CP1", &remove_duplicated_points<Atlas::CP1Product<2>>, "Remove duplicated points in CP1 point list.");
}