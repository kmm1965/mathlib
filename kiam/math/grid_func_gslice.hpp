#pragma once

#include "vector_grid_function.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
struct grid_func_gslice : math_object<grid_func_gslice<TAG, T> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> gf_type;
    typedef typename gf_type::proxy_type gf_proxy_type;
    typedef typename gf_type::reference reference;
    typedef typename gf_type::const_reference const_reference;

    struct index_type
    {
        index_type(std::gslice const& gsl) : gsl(gsl){
            assert(gsl.size().size() > 0);
            assert(gsl.size().size() == gsl.stride().size());
        }

        __DEVICE
        size_t size() const {
            std::valarray<size_t> sizes = gsl.size();
            return math_reduce(std::begin(sizes), std::end(sizes), size_t(1), multiplies<size_t>());
        }

        __DEVICE
        size_t operator()(size_t i) const {
            size_t result = gsl.start();
            const std::valarray<size_t> sizes = gsl.size(), strides = gsl.stride();
            size_t const dims = sizes.size();
            for (unsigned idx = 0; idx < dims && i > 0; ++idx){
                size_t const isize = sizes[idx];
                result += i % isize * strides[idx];
                i /= isize;
            }
            return result;
        }

        std::gslice const gsl;
    };

    grid_func_gslice(gf_type& gf, std::gslice const& gsl) : gf_proxy(gf), index(gsl)
    {
        assert(index(0) < gf.size());
        assert(index(index.size() - 1) < gf.size());
    }

    __DEVICE __HOST
    size_t size() const {
        return index.size();
    }

    __DEVICE
    reference operator[](size_t i){
        assert(i < size());
        return gf_proxy[index(i)];
    }

    __DEVICE
    const_reference operator[](size_t i) const {
        assert(i < size());
        return gf_proxy[index(i)];
    }

    struct grid_func_gslice_assign_value_closure
    {
        grid_func_gslice_assign_value_closure(gf_proxy_type& gf_proxy, value_type const& value, index_type const& index) :
            gf_proxy(gf_proxy), value(value), index(index){}

        __DEVICE
        void operator()(size_t i){
            gf_proxy[index(i)] = value;
        }

    private:
        gf_proxy_type gf_proxy;
        value_type const value;
        index_type const index;
    };

    void operator=(value_type const& value){
        grid_func_gslice_assign_value_closure closure(gf_proxy, value, index);
        default_executor<tag_type>()(closure, size());
    }

    template<typename GEXP>
    struct grid_func_gslice_assign_gexp_closure
    {
        grid_func_gslice_assign_gexp_closure(gf_proxy_type& gf_proxy, GRID_EXPR(GEXP) const& gexp, index_type const& index) :
            gf_proxy(gf_proxy), gexp_proxy(gexp.get_proxy()), index(index){}

        __DEVICE
        void operator()(size_t i){
            size_t const ind = index(i);
            gf_proxy[ind] = gexp_proxy[ind];
        }

    private:
        gf_proxy_type gf_proxy;
        typename GEXP::proxy_type const gexp_proxy;
        index_type const index;
    };

    template<typename GEXP>
    typename std::enable_if<std::is_same<typename GEXP::tag_type, TAG>::value>::type
    operator=(GRID_EXPR(GEXP) const& gexp){
        grid_func_gslice_assign_gexp_closure<GEXP> closure(gf_proxy, gexp, index);
        default_executor<tag_type>()(closure, size());
    }

//private:
    gf_proxy_type gf_proxy;
    index_type const index;
};

template<typename TAG, typename T>
grid_func_gslice<TAG, T> vector_grid_function<TAG, T>::operator[](std::gslice const& gsl){
    return grid_func_gslice<TAG, T>(*this, gsl);
}

_KIAM_MATH_END
