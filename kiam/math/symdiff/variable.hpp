#pragma once

#include "int_constant.hpp"

_SYMDIFF_BEGIN

template<unsigned N>
struct variable : expression<variable<N> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef int_constant<M == N> type;
    };

    template<unsigned M>
    typename diff_type<M>::type diff() const
    {
        return typename diff_type<M>::type();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return vars[N];
    }
};

_SYMDIFF_END
