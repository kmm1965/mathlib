#pragma once

#include "int_constant.hpp"
#include "mul_expression.hpp"
#include "../pow.hpp"

_SYMDIFF_BEGIN

template<class E, int N>
struct pow_expression;

template<int N, class E>
constexpr typename std::enable_if<N == 0, int_constant<1> >::type
pow(expression<E> const& e){
    return int_constant<0>();
}

template<int N, class E>
constexpr typename std::enable_if<N == 1, const E& >::type
pow(expression<E> const& e){
    return e();
}

template<int N, class E>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N> >::type
pow(expression<E> const& e);

template<int N, class E, int M>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N * M> >::type
pow(const pow_expression<E, M>& e);

template<typename E>
struct is_pow_expression : std::false_type {};

template<typename E, int N>
struct is_pow_expression<pow_expression<E, N> > : std::true_type {};

template<typename E>
struct pow_expression_expr_type {
    typedef void type;
};

template<typename E, int N>
struct pow_expression_expr_type<pow_expression<E, N> > {
    typedef E type;
};

template<typename E>
struct pow_expression_index : std::integral_constant<int, 0>{};

template<typename E, int N>
struct pow_expression_index<pow_expression<E, N> > : std::integral_constant<int, N> {};

template<class E, int N>
struct pow_expression_type
{
    typedef std::conditional_t<
        N == 0, int_constant<1>,
        std::conditional_t<
            N == 1, E,
            std::conditional_t<
                is_pow_expression<E>::value,
                pow_expression<typename pow_expression_expr_type<E>::type, N * pow_expression_index<E>::value>,
                pow_expression<E, N>
            >
        >
    > type;
};

template<class E, int N>
using pow_expression_t = typename pow_expression_type<E, N>::type;

template<class E, int N>
struct pow_expression : expression<pow_expression<E, N> >
{
    typedef pow_expression type;

    template<unsigned M>
    struct diff_type {
        typedef mul_expression_t<
            mul_expression_t<
                int_constant<N>,
                typename pow_expression_type<E, N - 1>::type
            >,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr pow_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return int_constant<N>() * _SYMDIFF::pow<N - 1>(e) * e.template diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return _KIAM_MATH::math_pow<N>(e(vars));
    }

private:
    E const e;
};

template<int N, class E>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N> >::type
pow(expression<E> const& e){
    return pow_expression<E, N>(e);
}

template<int N, class E, int M>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N * M> >::type
pow(const pow_expression<E, M>& e) {
    return pow_expression<E, N * M>(e.expr());
}

_SYMDIFF_END
