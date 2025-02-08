#pragma once

#include "int_constant.hpp"

_SYMDIFF_BEGIN

template<typename VT>
struct scalar : expression<scalar<VT> >
{
    typedef VT value_type;

    template<unsigned M>
    struct diff_type
    {
        typedef int_constant<0> type;
    };

    constexpr scalar(const value_type &value) : value(value){}

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return typename diff_type<M>::type();
    }

    template<typename T, size_t _Size>
    constexpr value_type operator()(std::array<T, _Size> const&) const {
        return value;
    }

    const value_type value;
};

typedef scalar<int> int_scalar;
typedef scalar<float> float_scalar;
typedef scalar<double> double_scalar;

template<class E>
struct is_scalar : std::false_type{};

template<typename VT>
struct is_scalar<scalar<VT> > : std::true_type{};

template<class E>
constexpr bool is_scalar_v = is_scalar<E>::value;

template<typename T>
constexpr scalar<T> _(T const& val) {
    return scalar<T>(val);
}

_SYMDIFF_END
