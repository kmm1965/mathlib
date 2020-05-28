#pragma once

#include "vector_grid_function.hpp"
#include "stride_iterator.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
struct simple_slice : math_object<simple_slice<TAG, T> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> grid_function_type;
    typedef typename grid_function_type::proxy_type gf_proxy_type;
    typedef typename grid_function_type::reference reference;
    typedef typename grid_function_type::const_reference const_reference;

    typedef stride_iterator<typename gf_proxy_type::iterator> iterator;
    typedef const iterator const_iterator;

    struct index_type
    {
        index_type(size_t init_index, size_t stride) : init_index(init_index), stride(stride){}

        __DEVICE
        size_t operator()(size_t i) const {
            return init_index + i * stride;
        }

        const size_t init_index, stride;
    };

    simple_slice(grid_function_type &gf, size_t init_index, size_t stride, size_t count) :
        gf_proxy(gf), index(init_index, stride), count(count)
    {
        assert(init_index < gf.size());
        assert(stride > 0 && stride < gf.size());
        assert(init_index + (count - 1) * stride < gf.size());
    }

    __DEVICE
    size_t size() const {
        return count;
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

    __DEVICE
    iterator begin(){
        return iterator(gf_proxy.begin() + index(0), (int)index.stride);
    }

    __DEVICE
    iterator end(){
        return iterator(gf_proxy.begin() + index(count), (int)index.stride);
    }

    __DEVICE
    const_iterator begin() const {
        return const_cast<simple_slice&>(*this).begin();
    }

    __DEVICE
    const_iterator end() const {
        return const_cast<simple_slice&>(*this).end();
    }

    __DEVICE
    const_iterator cbegin() const {
        return begin();
    }

    __DEVICE
    const_iterator cend() const {
        return end();
    }

    __DEVICE
    void operator=(value_type const& value){
        MATH_FILL(begin(), end(), value);
    }
    
    template<typename EO>
    struct simple_slice_executor_closure
    {
        simple_slice_executor_closure(gf_proxy_type& gf_proxy, EOBJ(EO) const& eobj, index_type const& index) :
            gf_proxy(gf_proxy), eobj_proxy(eobj.get_proxy()), index(index){}

        void operator[](size_t i){
            size_t const ind = index(i);
            gf_proxy[ind] = eobj_proxy[ind];
        }

    private:
        gf_proxy_type gf_proxy;
        typename EO::proxy_type const eobj_proxy;
        index_type const index;
    };

    template<typename EO>
    void operator=(EOBJ(EO) const& eobj){
        static_assert(std::is_same<typename EO::tag_type, TAG>::value, "Tag types should be the same");
        simple_slice_executor_closure<EO> closure(gf_proxy, eobj, index);
        default_executor<tag_type>()(closure, size());
    }

private:
    gf_proxy_type gf_proxy;
    index_type const index;
    size_t const count;

};

template<typename TAG, typename T>
simple_slice<TAG, T> get_simple_slice(vector_grid_function<TAG, T> &gf, size_t init_index, size_t stride, size_t count){
    return simple_slice<TAG, T>(gf, init_index, stride, count);
}

_KIAM_MATH_END
