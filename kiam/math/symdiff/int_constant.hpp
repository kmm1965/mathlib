#pragma once

#include "expression.hpp"

_SYMDIFF_BEGIN

template<int N>
struct int_constant : expression<int_constant<N> >
{
	static const int value = N;

	template<unsigned M>
	struct diff_type
	{
		typedef int_constant<0> type;
	};

	template<unsigned M>
	typename diff_type<M>::type diff() const {
		return typename diff_type<M>::type();
	}

	template<typename T, size_t _Size>
    constexpr int operator()(const std::array<T, _Size> &vars) const {
		return value;
	}
};

template<class E>
struct is_int_constant : std::false_type{};

template<int N>
struct is_int_constant<int_constant<N> > : std::true_type{};

template<class E>
struct int_constant_value : std::integral_constant<int, 0>{};

template<int N>
struct int_constant_value<int_constant<N> > : std::integral_constant<int, N>{};

_SYMDIFF_END
