#pragma once

#include "symdiff_def.h"

_SYMDIFF_BEGIN

template<typename T, size_t N>
struct square_matrix : std::array<T, N * N>
{
    typedef std::array<T, N * N> super;
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;

    constexpr size_t index(size_t i, size_t j) const
    {
        assert(i < N);
        assert(j < N);
        return i * N + j;
    }

    constexpr reference operator()(size_t i, size_t j){
        return super::operator[](index(i, j));
    }

    constexpr const_reference operator()(size_t i, size_t j) const {
        return super::operator[](index(i, j));
    }
};

template<typename T, size_t N>
std::ostream& operator<<(std::ostream &o, const square_matrix<T, N> &m)
{
    o << '{' << std::endl;;
    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < N; ++j)
            o << m(i, j) << ' ';
        o << std::endl;
    }
    return o << '}';
}

_SYMDIFF_END
