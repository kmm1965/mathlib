#pragma once

#include "../meta_loop.hpp"
#include "expression.hpp"

_SYMDIFF_BEGIN

template<class E, size_t N>
struct grad_closure
{
    constexpr grad_closure(expression<E> const& expr, std::array<sd_any, N> &result) :
        expr(expr()), result(result){}

    template<unsigned I>
    constexpr void apply(){
        result[I] = expr.template diff<I>();
    }

private:
    E const& expr;
    std::array<sd_any, N> &result;
};

template<size_t N, class E>
constexpr std::array<sd_any, N> grad(expression<E> const& expr)
{
    std::array<sd_any, N> result;
    grad_closure<E, N> closure(expr, result);
    meta_loop<N>(closure);
    return result;
}

template<class E, size_t N, class E2>
struct mul_expr_grad_closure
{
    constexpr mul_expr_grad_closure(expression<E2> const& expr, std::array<sd_any, N> const& vgrad, std::array<sd_any, N> &result) :
        expr(expr()), vgrad(vgrad), result(result) {}

    template<unsigned I>
    constexpr void apply(){
        result[I] = expr * ANY_CAST(typename E::template diff_type<I>::type, vgrad[I]);
    }

private:
    E2 const& expr;
    std::array<sd_any, N> const& vgrad;
    std::array<sd_any, N> &result;
};

template<class E, size_t N, class E2>
constexpr std::array<sd_any, N> mult(expression<E2> const& expr, std::array<sd_any, N> const& vgrad)
{
    std::array<sd_any, N> result;
    mul_expr_grad_closure<E, N, E2> closure(expr, vgrad, result);
    meta_loop<0, N>()(closure);
    return result;
}

template<class E, typename T, size_t N>
struct eval_grad_closure
{
    constexpr eval_grad_closure(std::array<sd_any, N> const& gradx,
        val_array<T, N> const& x, val_array<T, N> &result) :
        gradx(gradx), x(x), result(result){}

    template<unsigned I>
    constexpr void apply(){
        result[I] = ANY_CAST(typename E::template diff_type<I>::type const&, gradx[I])(x);
    }

private:
    std::array<sd_any, N> const& gradx;
    val_array<T, N> const& x;
    val_array<T, N> &result;
};

template<class E, typename T, size_t N>
constexpr val_array<T, N> eval_grad(std::array<sd_any, N> const& gradx, val_array<T, N> const& x)
{
    val_array<T, N> result;
    eval_grad_closure<E, T, N> closure(gradx, x, result);
    meta_loop<N>(closure);
    return result;
}

_SYMDIFF_END
