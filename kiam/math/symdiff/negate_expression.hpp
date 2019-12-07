#pragma once

#include "scalar.hpp"

_SYMDIFF_BEGIN

template<class E>
struct negate_expression;

template<class E>
struct negate_expression_type
{
	typedef typename std::conditional<
		is_int_constant<E>::value,
		int_constant<-int_constant_value<E>::value>,
		typename std::conditional<is_scalar<E>::value, E, negate_expression<E> >::type
	>::type type;
};

template<class E>
struct negate_expression : expression<negate_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename negate_expression_type<typename E::template diff_type<M>::type>::type type;
	};

    constexpr negate_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
		return -e.diff<M>();
	}

	template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
		return -e(vars);
	}

private:
	const E e;
};

template<class E>
constexpr negate_expression<E>
operator-(const expression<E>& e){
	return negate_expression<E>(e);
}

template<int N>
constexpr int_constant<-N>
operator-(const int_constant<N>&){
	return int_constant<-N>();
}

template<typename VT>
constexpr scalar<VT>
operator-(const scalar<VT> &e){
	return scalar<VT>(-e.value);
}

_SYMDIFF_END
