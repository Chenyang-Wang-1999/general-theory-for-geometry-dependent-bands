/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-16
 * @lastModified  2025-03-31
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */

#ifndef ATLAS_HPP
#define ATLAS_HPP

#include "../basic_defs.h"
#include <vector>
#include <algorithm>
#include <array>
#include <math.h>
#include <complex>
#include <map>
#include <list>
#include <eigen3/Eigen/Core>


using namespace std;


namespace Atlas{
/*
    PointData: a structure to store the data of point
*/
template<typename ScalarType, IndexType dim, ChartIdType chart_id_dim>
struct PointData
{
    std::array<ChartIdType, chart_id_dim> chart_labels;
    std::array<ScalarType, dim> coords;
};

/*
    PointView: a viewer to access each point
*/
template<typename Derived, typename ScalarType, IndexType Dim, 
    ChartIdType ChartIdDim>
class PointView
{
public:
    // Wrong!!
    // const static ChartIdType chart_id_dim = Derived::chart_id_dim;
    // const static IndexType dim = Derived::dim;
    typedef PointData<ScalarType, Dim, ChartIdDim> PointDataType;
    typedef std::array<ChartIdType, ChartIdDim> ChartLabelType;
    typedef std::array<ScalarType, Dim> CoordType;
    static const IndexType dim = Dim, chart_id_dim = ChartIdDim;
    ScalarType *coord_ptr = NULL;
    ChartIdType *chart_label_ptr = NULL;

    // Constructor
    PointView(){} // default constructor
    PointView(ScalarType * coord_ptr, ChartIdType * chart_label_ptr)
    {
        this->coord_ptr = coord_ptr;
        this->chart_label_ptr = chart_label_ptr;
    }
    // PointView(PointDataType * point_data_ptr)
    // {
    //     this->coord_ptr = point_data_ptr->coords.data();
    //     this->chart_label_ptr = point_data_ptr->chart_labels.data();
    // }
    // PointView(PointDataType & point_data) // addresses of coord.data my change
    //     {
    //         STDERR << point_data.coords.data() << '\n';
    //         this->coord_ptr = point_data.coords.data();
    //         this->chart_label_ptr = point_data.chart_labels.data();
    //     }

    // methods
    void to_chart(const ChartLabelType & new_chart_label){
        static_cast<Derived*>(this)->to_chart(new_chart_label);
    }
    RealType distance_sq(const Derived & another){
        return static_cast<Derived*>(this)->distance_sq(another);
    }
    void to_standard(){
        static_cast<Derived*>(this)->to_standard();
    }
    RealType distance(const Derived & another)
    {
        return sqrt(distance_sq(another));
    }

    // generate new point and reinitialize the viewer to that point
    void generate_new_point(PointDataType & point_data, const CoordType coords, const ChartLabelType chart_labels)
    {
        for(ChartIdType chart_id = 0; chart_id < chart_id_dim; chart_id ++)
        {
            point_data.chart_labels[chart_id] = chart_labels[chart_id];
        }

        for(IndexType curr_dim = 0; curr_dim < dim; curr_dim ++)
        {
            point_data.coords[curr_dim] = coords[curr_dim];
        }
        reinit_data(point_data.coords.data(), point_data.chart_labels.data());
    }

    void copy_data(PointDataType & new_data)
    {
        for(ChartIdType chart_id = 0; chart_id < chart_id_dim; chart_id ++)
        {
            new_data.chart_labels[chart_id] = chart_label_ptr[chart_id];
        }
        for(IndexType curr_dim = 0; curr_dim < dim; curr_dim ++)
        {
            new_data.coords[curr_dim] = coord_ptr[curr_dim];
        }
    }

    void reinit_data(ScalarType *coord_ptr, ChartIdType * chart_label_ptr)
    {
        this->coord_ptr = coord_ptr;
        this->chart_label_ptr = chart_label_ptr;
    }

    void set_data(const CoordType & coord, const ChartLabelType & chart_labels)
    {
        for(IndexType curr_coord_id = 0; curr_coord_id < dim; curr_coord_id ++)
        {
            coord_ptr[curr_coord_id] = coord[curr_coord_id];
        }
        for(ChartIdType curr_chart_id = 0; curr_chart_id < chart_id_dim; curr_chart_id ++)
        {
            chart_label_ptr[curr_chart_id] = chart_labels[curr_chart_id];
        }
    }
};


template<typename Derived, typename ScalarType,
    IndexType Dim, ChartIdType ChartIdDim, IndexType AffineDim>
class ManifoldInAffine: public PointView<Derived, ScalarType, Dim, ChartIdDim>
{
public:
    const static IndexType dim = Dim;
    const static ChartIdType chart_id_dim = ChartIdDim;
    const static IndexType affine_dim = AffineDim;
    typedef PointData<ScalarType, Dim, ChartIdDim> PointDataType;
    typedef std::array<ChartIdType, ChartIdDim> ChartLabelType;
    typedef std::array<ScalarType, Dim> CoordType;

    ManifoldInAffine(): PointView<
        Derived, ScalarType, Dim, ChartIdDim>(){} // default constructor

    ManifoldInAffine(
        ScalarType * coord_ptr, 
        ChartIdType * chart_label_ptr
    ):
        PointView<Derived, ScalarType, Dim, ChartIdDim>(
            coord_ptr, chart_label_ptr){}

    // methods
    void embed_into_affine(
        array<RealType, affine_dim> & affine_coord) const
    {
        static_cast<const Derived*>(this)->embed_into_affine(affine_coord);
    }

    void set_from_affine(const array<RealType, affine_dim> & affine_coord)
    {
        static_cast<Derived*>(this)->set_from_affine(affine_coord);
    }

    RealType distance_sq(const Derived & another)
    {
        array<RealType, affine_dim> affine_coord_1, affine_coord_2;
        embed_into_affine(affine_coord_1);
        another.embed_into_affine(affine_coord_2);
        RealType d_sq = 0.0;
        for(IndexType curr_dim = 0; curr_dim < affine_dim; curr_dim ++)
        {
            d_sq += pow(affine_coord_1[curr_dim] - affine_coord_2[curr_dim], 2);
        }
        return d_sq;
    }

    void linear_interp(const Derived & another, Derived & result, RealType s) 
    {
        array<RealType, affine_dim> affine_coord_1, affine_coord_2;
        embed_into_affine(affine_coord_1);
        another.embed_into_affine(affine_coord_2);
        for(IndexType curr_dim = 0; curr_dim < affine_dim; curr_dim ++)
        {
            affine_coord_1[curr_dim] = (1 - s) * affine_coord_1[curr_dim] + s * affine_coord_2[curr_dim];
        }
        result.set_from_affine(affine_coord_1);
    }
};


template<typename PointViewType>
double distance(PointViewType point_1, PointViewType point_2)
{
    return point_1.distance(point_2);
}


template<typename PointViewType>
double get_distance(
    typename PointViewType::PointDataType point_data_1,
    typename PointViewType::PointDataType point_data_2)
{
    PointViewType viewer1, viewer2;
    viewer1.reinit_data(point_data_1.coords.data(), point_data_1.chart_labels.data());
    viewer2.reinit_data(point_data_2.coords.data(), point_data_2.chart_labels.data());
    return distance(viewer1, viewer2);
}



/* Special Manifolds */
template <IndexType Dim>
class RealN: public ManifoldInAffine<RealN<Dim>, RealType, Dim, 0, Dim>
{
public:
    typedef RealType ScalarType;
    typedef PointData<ScalarType, Dim, 0> PointDataType;
    typedef std::array<ChartIdType, 0> ChartLabelType;
    typedef std::array<ScalarType, Dim> CoordType;
    static const IndexType dim = Dim; 
    static const ChartIdType chart_id_dim = 0; 
    static const IndexType affine_dim = Dim;

    RealN(): ManifoldInAffine<RealN<Dim>, RealType, Dim, 0, Dim>(){} // default constructor
    RealN(RealType * coord_ptr, ChartIdType * chart_label_ptr):
        ManifoldInAffine<RealN<Dim>, RealType, Dim, 0, Dim>(
            coord_ptr, chart_label_ptr){}


    void to_chart(const ChartLabelType & new_chart_label){}

    void embed_into_affine(array<RealType, Dim> & affine_coord) const
    {
        copy(this->coord_ptr, this->coord_ptr + dim, affine_coord.begin());
    }

    void set_from_affine(const array<RealType, Dim> & affine_coord)
    {
        copy(affine_coord.begin(), affine_coord.end(), this->coord_ptr);
    }
 
    void to_standard(){}
};


template<IndexType Dim>
class ComplexN: public ManifoldInAffine<ComplexN<Dim>, ComplexType, Dim, 0, 2 * Dim>
{
public:
    static const IndexType dim = Dim; 
    static const ChartIdType chart_id_dim = 0; 
    static const IndexType affine_dim = 2 * Dim;
    typedef ComplexType ScalarType;
    typedef PointData<ScalarType, dim, chart_id_dim> PointDataType;
    typedef std::array<ChartIdType, chart_id_dim> ChartLabelType;
    typedef std::array<ScalarType, dim> CoordType;

    ComplexN(): ManifoldInAffine<ComplexN<Dim>, ComplexType, Dim, 0, 2 * Dim>(){} // default constructor
    ComplexN(ComplexType * coord_ptr, ChartIdType * chart_label_ptr):
        ManifoldInAffine<ComplexN<Dim>, ComplexType, Dim, 0, 2 * Dim>(coord_ptr, chart_label_ptr){}


    void to_chart(const ChartLabelType & new_chart_label){}

    void embed_into_affine(array<RealType, 2 * Dim> & affine_coord) const
    {
        for(IndexType curr_dim = 0; curr_dim < Dim; curr_dim ++)
        {
            affine_coord[2 * curr_dim] = this -> coord_ptr[curr_dim].real();
            affine_coord[2 * curr_dim + 1] = this -> coord_ptr[curr_dim].imag();
        }
    }

    void set_from_affine(const array<RealType, 2 * Dim> & affine_coord)
    {
        for(IndexType curr_dim = 0; curr_dim < Dim; curr_dim ++)
        {
            this -> coord_ptr[curr_dim] = ComplexType(affine_coord[2 * curr_dim], affine_coord[2 * curr_dim + 1]);
        }
    }

    void to_standard(){}
};


class CP1: public ManifoldInAffine<CP1, ComplexType, 1, 1, 3>
{
public:
    typedef ComplexType ScalarType;
    const static ChartIdType chart_id_dim = 1;
    const static IndexType dim = 1;
    static const IndexType affine_dim = 3;
    typedef PointData<ScalarType, dim, chart_id_dim> PointDataType;
    typedef std::array<ChartIdType, chart_id_dim> ChartLabelType;
    typedef std::array<ScalarType, dim> CoordType;


    CP1(): ManifoldInAffine<CP1, ComplexType, 1, 1, 3>(){} // default constructor
    CP1(ComplexType * coord_ptr, ChartIdType * chart_label_ptr):
        ManifoldInAffine<CP1, ComplexType, 1, 1, 3>(coord_ptr, chart_label_ptr) {}

    void to_chart(const ChartLabelType & new_chart_label)
    {
        if(this -> chart_label_ptr[0] != new_chart_label[0])
        {
            this -> chart_label_ptr[0] = new_chart_label[0];
            this -> coord_ptr[0] = 1.0/(this -> coord_ptr[0]);
        }
    }

    void embed_into_affine(array<RealType, 3>& coords_3d) const
    {
        // embed the CP1 into complex sphere in cartesian coordinates
        if(this -> chart_label_ptr[0] == 0)
        {
            // north
            ComplexType beta = this -> coord_ptr[0];
            RealType beta_norm_sq = pow(beta.real(), 2) + pow(beta.imag(), 2);
            coords_3d[0] = 2*beta.real()/(beta_norm_sq + 1);
            coords_3d[1] = 2*beta.imag()/(beta_norm_sq + 1);
            coords_3d[2] = (beta_norm_sq - 1)/(beta_norm_sq + 1);
        }
        else
        {
            // south
            ComplexType gamma = this -> coord_ptr[0];
            RealType gamma_norm_sq = pow(gamma.real(), 2) + pow(gamma.imag(), 2);
            coords_3d[0] = 2*gamma.real()/(gamma_norm_sq + 1);
            coords_3d[1] = - 2*gamma.imag()/(gamma_norm_sq + 1);
            coords_3d[2] = (1-gamma_norm_sq)/(1+gamma_norm_sq);
        }
    }

    /* Get the z component of the 3d coordinates */
    RealType get_z_coord()
    {
        if(this -> chart_label_ptr[0] == 0)
        {
            // north
            ComplexType beta = this -> coord_ptr[0];
            RealType beta_norm_sq = pow(beta.real(), 2) + pow(beta.imag(), 2);
            return (beta_norm_sq - 1)/(beta_norm_sq + 1);
        }
        else
        {
            // south
            ComplexType gamma = this -> coord_ptr[0];
            RealType gamma_norm_sq = pow(gamma.real(), 2) + pow(gamma.imag(), 2);
            return (1-gamma_norm_sq)/(1+gamma_norm_sq);
        }
    }

    /* set CP1 point from 3d coordinates, in standard form */
    void set_from_affine(const std::array<RealType, 3> & coords_3d)
    {
        if(coords_3d[2] > 0)
        {
            // gamma
            this->chart_label_ptr[0] = 1;
            this->coord_ptr[0] = ComplexType(
                coords_3d[0]/(coords_3d[2] + 1), 
                - coords_3d[1]/(coords_3d[2] + 1)
            );
        }
        else
        {
            // beta
            this->chart_label_ptr[0] = 0;
            this->coord_ptr[0] = ComplexType(
                coords_3d[0]/(1 - coords_3d[2]),
                coords_3d[1]/(1 - coords_3d[2])
            );
        }
    }

    void to_standard(){
        RealType norm_sq = pow(this -> coord_ptr[0].real(), 2) + pow(this -> coord_ptr[0].imag(), 2);
        if(norm_sq > 1)
        {
            to_chart({!(this->chart_label_ptr)[0]});
        }
    }

};



template<typename Derived1, typename Derived2>
class AffineDirectProduct: 
    public ManifoldInAffine<
        AffineDirectProduct<Derived1, Derived2>,
        typename Derived1::ScalarType,
        Derived1::dim + Derived2::dim,
        Derived1::chart_id_dim + Derived2::chart_id_dim,
        Derived1::affine_dim + Derived2::affine_dim
    >
{
public:
    typedef typename Derived1::ScalarType ScalarType;
    const static ChartIdType chart_id_dim = 
        Derived1::chart_id_dim + Derived2::chart_id_dim;
    const static IndexType dim = 
        Derived1::dim + Derived2::dim;
    static const IndexType affine_dim = 
        Derived1::affine_dim + Derived2::affine_dim;
    typedef PointData<ScalarType, dim, chart_id_dim> PointDataType;
    typedef std::array<ChartIdType, chart_id_dim> ChartLabelType;
    typedef std::array<ScalarType, dim> CoordType;

    AffineDirectProduct():
        ManifoldInAffine<
            AffineDirectProduct<Derived1, Derived2>,
            ScalarType,
            Derived1::dim + Derived2::dim,
            Derived1::chart_id_dim + Derived2::chart_id_dim,
            Derived1::affine_dim + Derived2::affine_dim
        >(){} // default constructor
    AffineDirectProduct(ScalarType * coord_ptr, ChartIdType * chart_label_ptr):
        ManifoldInAffine<
            AffineDirectProduct<Derived1, Derived2>,
            ScalarType,
            Derived1::dim + Derived2::dim,
            Derived1::chart_id_dim + Derived2::chart_id_dim,
            Derived1::affine_dim + Derived2::affine_dim
        >(coord_ptr, chart_label_ptr){}

    Derived1 get_left()
    {
        return Derived1(this -> coord_ptr, this -> chart_label_ptr);
    }

    Derived2 get_right()
    {
        return Derived2(this -> coord_ptr + Derived1::dim, this -> chart_label_ptr + Derived1::chart_id_dim);
    }

    const Derived1 get_left() const
    {
        return Derived1(this -> coord_ptr, this -> chart_label_ptr);
    }

    const Derived2 get_right() const
    {
        return Derived2(this -> coord_ptr + Derived1::dim, this -> chart_label_ptr + Derived1::chart_id_dim);
    }

    void to_chart(const ChartLabelType & new_chart_label)
    {
        typename Derived1::ChartLabelType new_chart_label_left;
        typename Derived2::ChartLabelType new_chart_label_right;
        copy(new_chart_label.begin(), new_chart_label.begin() + Derived1::chart_id_dim, new_chart_label_left.begin());
        copy(new_chart_label.begin() + Derived1::chart_id_dim, new_chart_label.end(), new_chart_label_right.begin());
        get_left().to_chart(new_chart_label_left);
        get_right().to_chart(new_chart_label_right);
    }

    void embed_into_affine(array<RealType, Derived1::affine_dim + Derived2::affine_dim> & affine_coord) const
    {
        array<RealType, Derived1::affine_dim> affine_coord_left;
        array<RealType, Derived2::affine_dim> affine_coord_right;
        get_left().embed_into_affine(affine_coord_left);
        get_right().embed_into_affine(affine_coord_right);
        copy(affine_coord_left.begin(), affine_coord_left.end(), affine_coord.begin());
        copy(affine_coord_right.begin(), affine_coord_right.end(), affine_coord.begin() + Derived1::affine_dim);
    }

    void set_from_affine(const array<RealType, Derived1::affine_dim + Derived2::affine_dim> & affine_coord)
    {
        array<RealType, Derived1::affine_dim> affine_coord_left;
        array<RealType, Derived2::affine_dim> affine_coord_right;
        copy(affine_coord.begin(), affine_coord.begin() + Derived1::affine_dim, affine_coord_left.begin());
        copy(affine_coord.begin() + Derived1::affine_dim, affine_coord.end(), affine_coord_right.begin());
        get_left().set_from_affine(affine_coord_left);
        get_right().set_from_affine(affine_coord_right);
    }

    void to_standard()
    {
        get_left().to_standard();
        get_right().to_standard();
    }
};


template<typename Derived, IndexType N>
class AffineNProduct:
    public ManifoldInAffine<
        AffineNProduct<Derived, N>,
        typename Derived::ScalarType,
        Derived::dim * N,
        Derived::chart_id_dim * N,
        Derived::affine_dim * N
    >
{
public:
    const static ChartIdType chart_id_dim = 
        Derived::chart_id_dim * N;
    const static IndexType dim = 
        Derived::dim * N;
    static const IndexType affine_dim = 
        Derived::affine_dim * N;
    typedef typename Derived::ScalarType ScalarType;
    typedef PointData<ScalarType, dim, chart_id_dim> PointDataType;
    typedef std::array<ChartIdType, chart_id_dim> ChartLabelType;
    typedef std::array<ScalarType, dim> CoordType;

    AffineNProduct():
        ManifoldInAffine<AffineNProduct<Derived, N>, ScalarType, dim, chart_id_dim, affine_dim>(){} // default constructor
    AffineNProduct(ScalarType * coord_ptr, ChartIdType * chart_label_ptr):
        ManifoldInAffine<AffineNProduct<Derived, N>, ScalarType, dim, chart_id_dim, affine_dim>(coord_ptr, chart_label_ptr){}

    Derived get(IndexType index)
    {
        return Derived(this -> coord_ptr + index * Derived::dim, this -> chart_label_ptr + index * Derived::chart_id_dim);
    }

    const Derived get(IndexType index) const
    {
        return Derived(this -> coord_ptr + index * Derived::dim, this -> chart_label_ptr + index * Derived::chart_id_dim);
    }

    void to_chart(const ChartLabelType & new_chart_label)
    {
        typename Derived::ChartLabelType temp_chart_label;
        for(IndexType idx = 0; idx < N; idx ++)
        {
            copy(
                new_chart_label.begin() + idx * Derived::chart_id_dim, 
                new_chart_label.begin() + (idx + 1) * Derived::chart_id_dim, 
                temp_chart_label.begin()
            );
            get(idx).to_chart(temp_chart_label);
        }
    }

    void embed_into_affine(array<RealType, Derived::affine_dim * N> & affine_coord) const
    {
        array<RealType, Derived::affine_dim> temp_affine_coord;
        for(IndexType idx = 0; idx < N; idx ++)
        {
            get(idx).embed_into_affine(temp_affine_coord);
            copy(
                temp_affine_coord.begin(),
                temp_affine_coord.end(),
                affine_coord.begin() + idx * Derived::affine_dim
            );
        }
    }

    void set_from_affine(const array<RealType, Derived::affine_dim * N> & affine_coord)
    {
        array<RealType, Derived::affine_dim> temp_affine_coord;
        for(IndexType idx = 0; idx < N; idx ++)
        {
            copy(
                affine_coord.begin() + idx * Derived::affine_dim,
                affine_coord.begin() + (idx + 1) * Derived::affine_dim,
                temp_affine_coord.begin()
            );
            get(idx).set_from_affine(temp_affine_coord);
        }
    }

    void to_standard()
    {
        for(IndexType idx = 0; idx < N; idx ++)
        {
            get(idx).to_standard();
        }
    }
};


/* functions on the manifold */
// template<typename _PointViewType>
// class ManifoldFun
// {
//     typedef _PointViewType PointViewType;
//     virtual typename PointViewType::ScalarType operator()(typename PointViewType::PointDataType x) = 0;
// };

/* ultilities */

template<typename PointViewType>
void point_list_to_vec(const std::vector<typename PointViewType::PointDataType> point_list,
     std::vector<typename PointViewType::ScalarType>& loop_coords, 
        std::vector<ChartIdType>& loop_chart_labels)
    {
        IndexType point_list_length = point_list.size();
        loop_coords.clear(); loop_coords.resize(point_list_length * PointViewType::dim);
        loop_chart_labels.clear(); loop_chart_labels.resize(point_list_length * PointViewType::chart_id_dim);
        for(IndexType point_id = 0; point_id < point_list_length; point_id ++)
        {
            for(IndexType dim_id = 0; dim_id < PointViewType::dim; dim_id++)
            {
                loop_coords[point_id * PointViewType::dim + dim_id] = point_list[point_id].coords[dim_id];
            }
            for(IndexType chart_index_id = 0; chart_index_id < PointViewType::chart_id_dim; chart_index_id++)
            {
                loop_chart_labels[point_id * PointViewType::chart_id_dim + chart_index_id] = point_list[point_id].chart_labels[chart_index_id];
            }
        }
    }


template<typename PointViewType>
void point_list_to_vec(const std::list<typename PointViewType::PointDataType> point_list,
     std::vector<typename PointViewType::ScalarType>& loop_coords, 
        std::vector<ChartIdType>& loop_chart_labels)
{
        IndexType point_list_length = point_list.size();
        loop_coords.clear(); loop_coords.resize(point_list_length * PointViewType::dim);
        loop_chart_labels.clear(); loop_chart_labels.resize(point_list_length * PointViewType::chart_id_dim);
        IndexType point_id = 0;
        for(auto point_it:point_list)
        {
            for(IndexType dim_id = 0; dim_id < PointViewType::dim; dim_id++)
            {
                loop_coords[point_id * PointViewType::dim + dim_id] = point_it.coords[dim_id];
            }
            for(IndexType chart_index_id = 0; chart_index_id < PointViewType::chart_id_dim; chart_index_id++)
            {
                loop_chart_labels[point_id * PointViewType::chart_id_dim + chart_index_id] = point_it.chart_labels[chart_index_id];
            }
            point_id ++;
        }
}

/* algorithms */

/* 
    Given a point list, check whether the minimal distance is less than 
    a given tolerance
*/
template<typename PointViewType>
RealType check_min_distance(std::vector<typename PointViewType::PointDataType> &point_vec,
    RealType tol)
{
    // initialize min_pair and viewer
    PointViewType viewer1(point_vec[0].coords.data(), point_vec[0].chart_labels.data()),
         viewer2(point_vec[1].coords.data(), point_vec[1].chart_labels.data());
    RealType min_dist = 1e16;

    // traverse the whole list
    for(auto it1=point_vec.begin(); (it1+1)!=point_vec.end(); it1++)
    {
        for(auto it2 = it1+1; it2 != point_vec.end(); it2++)
        {
            viewer1.reinit_data((it1->coords.data()), (it1->chart_labels.data()));
            viewer2.reinit_data((it2->coords.data()), (it2->chart_labels.data()));
            RealType curr_dist = distance(viewer1, viewer2);
            if(curr_dist < tol)
            {
                return curr_dist;
            }
            if(curr_dist < min_dist)
            {
                min_dist = curr_dist;
            }
        }
    }
    return min_dist;
}

/*
    Check minimum distance between a point and a point list
*/
template<typename PointViewType>
RealType check_min_distance(typename PointViewType::PointDataType &point,
    std::vector<typename PointViewType::PointDataType> &point_vec,
    RealType tol)
    {
        PointViewType viewer1(point.coords.data(), point.chart_labels.data()), 
                viewer2(point_vec[0].coords.data(), point_vec[0].chart_labels.data());
        RealType min_dist = distance(viewer1, viewer2);

        // point_vec has a single point
        if(point_vec.size() == 1)
        {
            return min_dist;
        }

        for(auto it = point_vec.begin()+1; it!= point_vec.end(); it++)
        {
            viewer2.reinit_data((it->coords.data()), (it->chart_labels.data()));
            RealType curr_dist = distance(viewer1, viewer2);
            if(curr_dist < tol)
            {
                return curr_dist;
            }

            if(curr_dist < min_dist)
            {
                min_dist = curr_dist;
            }
        }
        return min_dist;
    }


/*
    Given a point and a point set, calculate the minimum distance and corresponding index
*/
template<typename PointViewType>
RealType calculate_min_distance(typename PointViewType::PointDataType &point,
    std::vector<typename PointViewType::PointDataType> &point_vec,
    IndexType & min_id)
    {
        PointViewType viewer1(point.coords.data(), point.chart_labels.data()),
                 viewer2(point_vec[0].coords.data(), point_vec[0].chart_labels.data());
        RealType min_dist = distance(viewer1, viewer2);
        min_id = 0;

        // point_vec has a single point
        if(point_vec.size() == 1)
        {
            return min_dist;
        }

        IndexType curr_id = 1;
        for(auto it = point_vec.begin()+1; it!= point_vec.end(); it++)
        {
            viewer2.reinit_data((it->coords.data()), (it->chart_labels.data()));
            RealType curr_dist = distance(viewer1, viewer2);
            if(curr_dist < min_dist)
            {
                min_dist = curr_dist;
                min_id = curr_id;
            }
            curr_id ++;
        }
        return min_dist;
    }

/* change the charts to standard */
template<typename PointViewType>
void batch_to_standard(std::vector<typename PointViewType::PointDataType> & point_list)
{
    for(auto it = point_list.begin(); it != point_list.end(); it++)
    {
        PointViewType viewer(it->coords.data(), it->chart_labels.data());
        viewer.to_standard();
    }
}

/*
    Get the unique elements of a point list
*/
template<typename PointViewType>
void get_unique_points(
    std::vector<typename PointViewType::PointDataType> & point_list,
    std::vector<typename PointViewType::PointDataType> & unique_list,
    RealType tol)
{
    unique_list.clear();
    if(point_list.size() == 0)
    {
        return;
    }

    unique_list.push_back(point_list[0]);

    // Traverse point_list, if the distance is greater than tol, append the 
    // point to unique_list
    for(auto it = point_list.begin() + 1; it != point_list.end(); it++)
    {
        RealType min_dist = check_min_distance<PointViewType>(*it, unique_list, tol);
        if(min_dist >= tol)
        {
            unique_list.push_back(*it);
        }
    }
}

template<typename PointViewType>
bool match_point_list_equal_length(std::vector<typename PointViewType::PointDataType> & curr_points, 
    std::vector<typename PointViewType::PointDataType> & new_points, 
    std::vector<typename PointViewType::PointDataType> & result)
{
    if(result.size() != curr_points.size())
    {
        result.clear();
        result.resize(curr_points.size());
    }
    if(new_points.size() == curr_points.size())
    {
        // for(auto it = curr_points.begin(); it != curr_points.end(); it++)
        for(IndexType curr_id = 0; curr_id < curr_points.size(); curr_id++)
        {
            IndexType curr_min_id;
            RealType curr_dist = calculate_min_distance<PointViewType>(
                curr_points[curr_id], new_points, curr_min_id 
            );

            // append to the new points
            result[curr_id] = new_points[curr_min_id];
            new_points.erase(new_points.begin() + curr_min_id);
        }
        return true;
    }
    else
    {
        return false;
    }
}


/*
    Match a point list with another
*/
template<typename PointViewType>
bool match_point_list(std::vector<typename PointViewType::PointDataType> & curr_points, 
    std::vector<typename PointViewType::PointDataType> & new_points, 
    std::vector<typename PointViewType::PointDataType> & result)
{
    RealType max_dist = 0.0;
    // initialize
    if(result.size() != curr_points.size())
    {
        result.clear();
        result.resize(curr_points.size());
    }

    // 1. equal length
    if(new_points.size() == curr_points.size())
    {
        return match_point_list_equal_length<PointViewType>(
            curr_points, new_points, result
        );
    }
    // 2. unequal length
    else if(new_points.size() < curr_points.size())
    {
        for(IndexType curr_id = 0; curr_id < new_points.size(); curr_id++)
        {
            IndexType curr_min_id;
            RealType curr_dist = calculate_min_distance<PointViewType>(
                curr_points[curr_id], new_points, curr_min_id 
            );

            // update max_dist
            if(max_dist < curr_dist)
            {
                max_dist = curr_dist;
            }

            // append to the new points
            result[curr_id] = new_points[curr_min_id];
            new_points.erase(new_points.begin() + curr_min_id);
        }
        for(IndexType curr_id = new_points.size(); curr_id < curr_points.size(); curr_id++)
        {
            IndexType curr_min_id;
            std::vector<typename PointViewType::PointDataType> stored_points(result.begin(), result.begin() + new_points.size());
            RealType curr_dist = calculate_min_distance<PointViewType>(
                curr_points[curr_id], stored_points, curr_min_id
            );

            // update max_dist
            if(max_dist < curr_dist)
            {
                max_dist = curr_dist;
            }
            result[curr_id] = stored_points[curr_min_id];
        }
        return true;
    }
    else
    {
        STDERR << "atlas.hpp match_point_list: more new points than old points\n";
        STDERR << "old points: " << curr_points.size() << '\n';
        STDERR << "new points: " << new_points.size() << '\n';
        // assert(0);
        return false;
    }
}

/* evaluate the value of each point in the point list*/
template<typename PointViewType, typename ResultType>
void batch_eval(std::vector<typename PointViewType::PointDataType> & point_list,
    std::vector<ResultType> & results, ResultType (*fun_call)(typename PointViewType::PointDataType))
{
    results.clear();
    results.resize(point_list.size());

    for(IndexType point_id = 0; point_id < point_list.size(); point_id ++)
    {
        results[point_id] = fun_call(point_list[point_id]);
    }
}


/* Get the interpolation parameter s from a given beta norm 
    Assuming: |p1z| > |cos(theta)| > |p2z|
*/
RealType interp_CP1_points_given_norm(
    const CP1 &point1, const CP1 & point2,
    RealType norm_cos_theta, RealType s_ini = 0.5,
    RealType tol=1e-12, IndexType max_iter=10000
)
{
    // 1. Get 3D coordinates
    std::array<RealType,3> point1_3d, point2_3d;
    point1.embed_into_affine(point1_3d);
    point2.embed_into_affine(point2_3d);

    // 2. Calculate p1 and v
    Eigen::Vector3d v;
    Eigen::Map<Eigen::Vector3d> 
        p1(point1_3d.data()), 
        p2(point2_3d.data());
    v = p2 - p1;
    
    // 3. Iteratively calculate s
    RealType s_curr = s_ini, s_next = s_ini;
    for(IndexType j = 0; j < max_iter; j++)
    // for(IndexType j = 0; j < 100; j++) // debug
    {
        s_curr = s_next;
        // s_next = (p1 + s_curr * v).norm() * norm_cos_theta - p1(2);
        s_next = ((p1 + s_curr * v).norm() - 1) * norm_cos_theta + (norm_cos_theta - p1(2));
        s_next /= v(2);
        if(abs(s_next - s_curr) < tol)
        {
            return s_next;
        }
        // STDERR << '(' << s_next << ',' << (p1 + s_curr * v).norm() * norm_cos_theta - p1(2) << ',' << v(2) << "),";
    }
    // STDERR << '\n';
    STDERR << "interp_CP1_points_given_norm: Warning: iteration not converged (N > " << max_iter << ")\n";
    return s_next;

}

/* Linearly interp the point data */
template<typename PointViewType>
void linear_interp(
    typename PointViewType::PointDataType & point_data1, 
    typename PointViewType::PointDataType & point_data2, 
    typename PointViewType::PointDataType & result_data,
    RealType s 
)
{
    PointViewType 
        viewer1(point_data1.coords.data(), point_data1.chart_labels.data()),
        viewer2(point_data2.coords.data(), point_data2.chart_labels.data()),
        result_view(result_data.coords.data(), result_data.chart_labels.data());
    viewer1.linear_interp(viewer2, result_view, s);
}

} // namespace Atlas

#endif