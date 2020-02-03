#pragma once

#include "../meta_loop.hpp"
#include "expression.hpp"

_SYMDIFF_BEGIN

template<class E, size_t N>
struct grad_closure
{
    constexpr grad_closure(const expression<E> &expr, std::array<sd_any, N> &result) :
        expr(expr()), result(result){}

    template<unsigned I>
    constexpr void apply(){
        result[I] = expr.template diff<I>();
    }

private:
    const E &expr;
    std::array<sd_any, N> &result;
};

template<size_t N, class E>
constexpr std::array<sd_any, N> grad(const expression<E> &expr)
{
    std::array<sd_any, N> result;
    grad_closure<E, N> closure(expr, result);
    meta_loop<N>(closure);
    return result;
}

template<class E, size_t N, class E2>
struct mul_expr_grad_closure
{
    constexpr mul_expr_grad_closure(const expression<E2> &expr, const std::array<sd_any, N> &vgrad, std::array<sd_any, N> &result) :
        expr(expr()), vgrad(vgrad), result(result) {}

    template<unsigned I>
    constexpr void apply(){
        result[I] = expr * ANY_CAST(typename E::template diff_type<I>::type, vgrad[I]);
    }

private:
    const E2 &expr;
    const std::array<sd_any, N> &vgrad;
    std::array<sd_any, N> &result;
};

template<class E, size_t N, class E2>
constexpr std::array<sd_any, N> mult(const expression<E2> &expr, const std::array<sd_any, N> &vgrad)
{
    std::array<sd_any, N> result;
    mul_expr_grad_closure<E, N, E2> closure(expr, vgrad, result);
    meta_loop<0, N>()(closure);
    return result;
}

template<class E, typename T, size_t N>
struct eval_grad_closure
{
    constexpr eval_grad_closure(const std::array<sd_any, N> &gradx,
        const val_array<T, N> &x, val_array<T, N> &result) :
        gradx(gradx), x(x), result(result){}

    template<unsigned I>
    constexpr void apply(){
        result[I] = ANY_CAST(typename E::template diff_type<I>::type const&, gradx[I])(x);
    }

private:
    const std::array<sd_any, N> &gradx;
    const val_array<T, N> &x;
    val_array<T, N> &result;
};

template<class E, typename T, size_t N>
constexpr val_array<T, N> eval_grad(const std::array<sd_any, N> &gradx, const val_array<T, N> &x)
{
    val_array<T, N> result;
    eval_grad_closure<E, T, N> closure(gradx, x, result);
    meta_loop<N>(closure);
    return result;
}

_SYMDIFF_END
