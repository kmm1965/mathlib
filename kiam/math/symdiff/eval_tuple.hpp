#pragma once

#include "val_array.hpp"

_SYMDIFF_BEGIN

template<typename T, unsigned M, class... E>
struct eval_tuple_closure
{
	static const size_t N = std::tuple_size<std::tuple<E...> >::value;

    constexpr eval_tuple_closure(const std::tuple<E...> &tp, const val_array<T, M> &x,  val_array<T, N> &result) :
		tp(tp), x(x), result(result){}

	template<unsigned I>
    constexpr void apply(){
		result[I] = std::get<I>(tp)(x);
	}

private:
	const std::tuple<E...> &tp;
	const val_array<T, M> &x;
	val_array<T, N> &result;
};

template<typename T, unsigned M, class... E>
constexpr val_array<T, std::tuple_size<std::tuple<E...> >::value>
eval_tuple(const std::tuple<E...> &tp, const val_array<T, M> &x)
{
	const size_t N = std::tuple_size<std::tuple<E...> >::value;
	val_array<T, N> result;
	eval_tuple_closure<T, M, E...> closure(tp, x, result);
	meta_loop<N>(closure);
	return result;
}

_SYMDIFF_END
