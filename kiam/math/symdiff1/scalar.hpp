#pragma once

#include "int_constant.hpp"

_SYMDIFF1_BEGIN

template<typename VT>
struct scalar : expression<scalar<VT> >
{
	typedef VT value_type;

	typedef int_constant<0> diff_type;

    constexpr scalar(const value_type &value) : value(value){}

    constexpr diff_type diff() const
	{
		return diff_type();
	}

	template<typename T>
    constexpr value_type operator()(const T &) const {
		return value;
	}

    constexpr std::string to_string() const
    {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }
    
    const value_type value;
};

typedef scalar<int> int_scalar;
typedef scalar<float> float_scalar;
typedef scalar<double> double_scalar;

template<class E>
struct is_scalar : std::false_type {};

template<typename VT>
struct is_scalar<scalar<VT> > : std::true_type {};

template<typename T>
constexpr scalar<T> _(const T &val) {
	return scalar<T>(val);
}

_SYMDIFF1_END
