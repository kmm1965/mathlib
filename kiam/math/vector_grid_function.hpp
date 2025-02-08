#pragma once

#include <valarray>

#include "grid_function.hpp"
#include "math_vector.hpp"
#include "host_vector.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
struct simple_slice;

template<typename TAG, typename T>
struct grid_func_gslice;

template<typename TAG, typename T>
struct vector_grid_function_proxy;

template<typename TAG, typename T>
struct vector_grid_function : grid_function<TAG, vector_grid_function<TAG, T>, vector_grid_function_proxy<TAG, T> >, math_vector<T>
{
    typedef vector_grid_function type;
    typedef TAG tag_type;
    typedef T value_type;
    typedef math_vector<value_type> vector_type;
    typedef typename vector_type::reference reference;
    typedef typename vector_type::const_reference const_reference;
    typedef typename vector_type::pointer pointer;
    typedef typename vector_type::const_pointer const_pointer;
    typedef grid_function<tag_type, type, vector_grid_function_proxy<tag_type, value_type> > super;

    vector_grid_function(vector_grid_function const& other) : vector_type(other), m_local_size(other.m_local_size){}
    vector_grid_function(vector_grid_function&&) = delete;

    vector_grid_function(size_t size, size_t local_size = 0, value_type const& init = value_type()) :
        vector_type(size, init), m_local_size(local_size == 0 ? size : local_size){}

    vector_grid_function(std::initializer_list<T> il) : super(il){}

    void operator=(value_type const& value);
    void operator=(const vector_grid_function &other);

    void operator=(const HOST_VECTOR_T<value_type> &hvec)
    {
        assert(hvec.size() == vector_type::size());
        vector_type::operator=(hvec);
    }

    template<class GEXP>
    void operator&=(GRID_EXPR(GEXP) const& gexp);

    template<typename GEXP>
    void operator=(GRID_EXPR(GEXP) const& gexp);

    //template<typename GEXP>
    //void operator=(func_grid_expression<typename GEXP::tag_type, GEXP, typename GEXP::proxy_type> const& fgexp);

    void resize(size_t new_size, size_t local_size = 0, value_type const& init = value_type())
    {
        vector_type::resize(new_size, init);
        m_local_size = local_size > 0 ? local_size : new_size;
    }

    size_t local_size() const { return m_local_size; }

    reference operator[](size_t i){
        return vector_type::operator[](i);
    }

    const_reference operator[](size_t i) const {
        return vector_type::operator[](i);
    }

    void operator+=(value_type const& value);
    void operator-=(value_type const& value);
    void operator*=(value_type const& value);
    void operator/=(value_type const& value);

    void operator+=(const vector_grid_function &other);
    void operator-=(const vector_grid_function &other);
    void operator*=(const vector_grid_function &other);
    void operator/=(const vector_grid_function &other);

    template<class GEXP>
    std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value> operator+=(GRID_EXPR(GEXP) const& gexp);

    template<class GEXP>
    std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value> operator-=(GRID_EXPR(GEXP) const& gexp);

    template<class GEXP>
    std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value> operator*=(GRID_EXPR(GEXP) const& gexp);

    template<class GEXP>
    std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value> operator/=(GRID_EXPR(GEXP) const& gexp);

    simple_slice<tag_type, value_type> operator[](std::slice const& sl);
    grid_func_gslice<tag_type, value_type> operator[](std::gslice const& gsl);

    size_t m_local_size;
};

_KIAM_MATH_END

#define DECLARE_MATH_VECTOR_GRID_FUNCTION(name) \
    template<typename T> \
    using name##_vector_grid_function = _KIAM_MATH::vector_grid_function<name##_tag, T>

#include "vector_grid_function.inl"
