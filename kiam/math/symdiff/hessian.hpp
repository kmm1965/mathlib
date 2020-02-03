#pragma once

#include "../meta_loop.hpp"
#include "expression.hpp"
#include "square_matrix.hpp"

_SYMDIFF_BEGIN

template<class E, size_t I, size_t N>
struct hessian1_closure
{
    constexpr hessian1_closure(const expression<E> &expr, square_matrix<sd_any, N> &result) : expr(expr()), result(result){}

    template<unsigned J>
    constexpr void apply(){
        result(I, J) = expr.template diff<J>();
    }

private:
    const E &expr;
    square_matrix<sd_any, N> &result;
};

template<class E, size_t N>
struct hessian0_closure
{
    constexpr hessian0_closure(const expression<E> &expr, square_matrix<sd_any, N> &result) : expr(expr()), result(result){}

    template<unsigned I>
    constexpr void apply()
    {
        typedef typename E::template diff_type<I>::type diff_type;
        hessian1_closure<diff_type, I, N> closure(expr.template diff<I>(), result);
        meta_loop<N>(closure);
    }

private:
    const E &expr;
    square_matrix<sd_any, N> &result;
};

template<size_t N, class E>
constexpr square_matrix<sd_any, N> hessian(const expression<E> &expr)
{
    square_matrix<sd_any, N> result;
    hessian0_closure<E, N> closure(expr, result);
    meta_loop<N>(closure);
    return result;
}

template<class E, size_t I, typename T, size_t N>
struct eval_hessian1_closure
{
    constexpr eval_hessian1_closure(const square_matrix<sd_any, N> &hessianx,
        const val_array<T, N> &x, square_matrix<T, N> &result) :
        hessianx(hessianx), x(x), result(result){}

    template<unsigned J>
    constexpr void apply()
    {
        typedef typename E::template diff_type<I>::type::
            template diff_type<J>::type diff_type;
        result(I, J) = ANY_CAST(diff_type const&, hessianx(I, J))(x);
    }

private:
    const square_matrix<sd_any, N> &hessianx;
    const val_array<T, N> &x;
    square_matrix<T, N> &result;
};

template<class E, typename T, size_t N>
struct eval_hessian0_closure
{
    constexpr eval_hessian0_closure(const square_matrix<sd_any, N> &hessianx,
        const val_array<T, N> &x, square_matrix<T, N> &result) :
        hessianx(hessianx), x(x), result(result){}

    template<unsigned I>
    constexpr void apply()
    {
        eval_hessian1_closure<E, I, T, N> closure(hessianx, x, result);
        meta_loop<N>(closure);
    }

private:
    const square_matrix<sd_any, N> &hessianx;
    const val_array<T, N> &x;
    square_matrix<T, N> &result;
};

template<class E, typename T, size_t N>
constexpr square_matrix<T, N> eval_hessian(const square_matrix<sd_any, N> &hessianx, const val_array<T, N> &x)
{
    square_matrix<T, N> result;
    eval_hessian0_closure<E, T, N> closure(hessianx, x, result);
    meta_loop<N>(closure);
    return result;
}

_SYMDIFF_END
