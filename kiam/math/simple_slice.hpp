#pragma once

#include "vector_grid_function.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
struct simple_slice : math_object<simple_slice<TAG, T> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> gf_type;
    typedef typename gf_type::proxy_type gf_proxy_type;
    typedef typename gf_type::reference reference;
    typedef typename gf_type::const_reference const_reference;

    struct index_type
    {
        index_type(std::slice const& sl) : m_start(sl.start()), m_size(sl.size()) , m_stride(sl.stride()){}

        __DEVICE
        size_t size() const {
            return m_size;
        }

        __DEVICE
        size_t operator()(size_t i) const {
            return m_start + i * m_stride;
        }

        const size_t m_start, m_size, m_stride;
    };

    simple_slice(gf_type &gf, std::slice const& sl) : gf_proxy(gf), index(sl)
    {
        assert(index(0) < gf.size());
        assert(sl.stride() > 0 && sl.stride() < gf.size());
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

    struct simple_slice_assign_value_closure
    {
        simple_slice_assign_value_closure(gf_proxy_type& gf_proxy, value_type const& value, index_type const& index) :
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
        simple_slice_assign_value_closure closure(gf_proxy, value, index);
        default_executor<tag_type>()(closure, size());
    }
    
    template<typename GEXP>
    struct simple_slice_assign_gexp_closure
    {
        simple_slice_assign_gexp_closure(gf_proxy_type& gf_proxy, GRID_EXPR(GEXP) const& gexp, index_type const& index) :
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
        simple_slice_assign_gexp_closure<GEXP> closure(gf_proxy, gexp, index);
        default_executor<tag_type>()(closure, size());
    }

private:
    gf_proxy_type gf_proxy;
    index_type const index;
};

template<typename TAG, typename T>
simple_slice<TAG, T> vector_grid_function<TAG, T>::operator[](std::slice const& sl){
    return simple_slice<TAG, T>(*this, sl);
}

_KIAM_MATH_END
