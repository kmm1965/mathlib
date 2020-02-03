#pragma once

#include <tuple>

#include "../meta_loop.hpp"
#include "expression.hpp"
#include "square_matrix.hpp"

_SYMDIFF_BEGIN

template<class E, size_t I, size_t N>
struct jacobian1_closure
{
    constexpr jacobian1_closure(const expression<E> &e, square_matrix<sd_any, N> &result) : e(e()), result(result){}

    template<unsigned J>
    constexpr void apply(){
        result(I, J) = e.template diff<J>();
    }

private:
    const E &e;
    square_matrix<sd_any, N> &result;
};

template<class... E>
struct jacobian0_closure
{
    static const size_t N = std::tuple_size<std::tuple<E...> >::value;

    constexpr jacobian0_closure(const std::tuple<E...> &tp, square_matrix<sd_any, N> &result) :
        tp(tp), result(result){}

    template<unsigned I>
    constexpr void apply()
    {
        typedef typename std::tuple_element<I, std::tuple<E...>>::type expr_type;
        jacobian1_closure<expr_type, I, N> closure(std::get<I>(tp), result);
        meta_loop<N>(closure);
    }

private:
    const std::tuple<E...> &tp;
    square_matrix<sd_any, N> &result;
};

template<class... E>
constexpr square_matrix<sd_any, std::tuple_size<std::tuple<E...> >::value>
jacobian(const std::tuple<E...> &tp)
{
    const size_t N = std::tuple_size<std::tuple<E...> >::value;
    square_matrix<sd_any, N> result;
    jacobian0_closure<E...> closure(tp, result);
    meta_loop<N>(closure);
    return result;
}

template<class E, typename T, size_t I, size_t N>
struct eval_jacobian1_closure
{
    constexpr eval_jacobian1_closure(const square_matrix<sd_any, N> &Jac, const val_array<T, N> &x,
        square_matrix<T, N> &result) : Jac(Jac), x(x), result(result){}

    template<unsigned J>
    constexpr void apply()
    {
        typedef typename E::template diff_type<J>::type diff_type;
        result(I, J) = ANY_CAST(const diff_type&, Jac(I, J))(x);
    }

private:
    const square_matrix<sd_any, N> &Jac;
    const val_array<T, N> &x;
    square_matrix<T, N> &result;
};

template<typename T, class... E>
struct eval_jacobian0_closure
{
    static const size_t N  = std::tuple_size<std::tuple<E...> >::value;

    constexpr eval_jacobian0_closure(const square_matrix<sd_any, N> &Jac, const val_array<T, N> &x,
        square_matrix<T, N> &result) : Jac(Jac), x(x), result(result){}

    template<unsigned I>
    constexpr void apply()
    {
        typedef typename std::tuple_element<I, std::tuple<E...> >::type expr_type;
        eval_jacobian1_closure<expr_type, T, I, N> closure(Jac, x, result);
        meta_loop<N>(closure);
    }

private:
    const square_matrix<sd_any, N> &Jac;
    const val_array<T, N> &x;
    square_matrix<T, N> &result;
};

template<typename T, class... E>
constexpr square_matrix<T, std::tuple_size<std::tuple<E...> >::value>
eval_jacobian(const square_matrix<sd_any, std::tuple_size<std::tuple<E...> >::value> &Jac, const val_array<T, std::tuple_size<std::tuple<E...> >::value> &x)
{
    const size_t N = std::tuple_size<std::tuple<E...> >::value;
    square_matrix<T, N> result;
    eval_jacobian0_closure<T, E...> closure(Jac, x, result);
    meta_loop<N>(closure);
    return result;
}

template<typename T, class... E>
constexpr square_matrix<T, std::tuple_size<std::tuple<E...> >::value>
eval_jacobian(const std::tuple<E...>&tp, const val_array<T, std::tuple_size<std::tuple<E...> >::value> &x){
    return eval_jacobian<T, E...>(jacobian(tp), x);
}

_SYMDIFF_END
