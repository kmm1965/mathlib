#pragma once

#include "pow_expression.hpp"

_SYMDIFF1_BEGIN

template<class E1, class E2>
struct div_expression;

template<class E1, class E2>
struct div_expression_type
{
    typedef std::conditional_t<
        std::is_same_v<E1, int_constant<0> >, int_constant<0>,
        std::conditional_t<std::is_same_v<E2, int_constant<0> >, void,
            std::conditional_t<std::is_same_v<E2, int_constant<1> >, E1,
                std::conditional_t<
                    is_scalar_v<E1> && (std::is_same_v<E1, E2> || is_int_constant_v<E2>), E1,
                    std::conditional_t<is_int_constant_v<E1> && is_scalar_v<E2>, E2, div_expression<E1, E2> >
                >
            >
        >
    > type;
};

template<class E1, class E2>
using div_expression_t = typename div_expression_type<E1, E2>::type;

template<class E1, class E2>
std::enable_if_t<is_int_constant_v<E2>, div_expression_t<typename E1::diff_type, E2> >
div_diff(E1 const& e1, E2 const& e2){
    return e1.diff() / e2;
}

template<class E1, class E2>
std::enable_if_t<!is_int_constant_v<E2>,
    div_expression_t<
        additive_expression_t<
            mul_expression_t<typename E1::diff_type, E2>,
            std::minus<double>,
            mul_expression_t<E1, typename E2::diff_type>
        >,
        pow_expression_t<E2, 2>
    >
> div_diff(E1 const& e1, E2 const& e2){
    return (e1.diff() * e2 - e1 * e2.diff()) / kiam::math::symdiff1::pow<2>(e2);
}

template<class E1, class E2>
struct div_expression : expression<div_expression<E1, E2> >
{
    typedef std::conditional_t<is_int_constant_v<E2>,
        div_expression_t<typename E1::diff_type, E2>,
        div_expression_t<
            additive_expression_t<
                mul_expression_t<typename E1::diff_type, E2>,
                std::minus<double>,
                mul_expression_t<E1, typename E2::diff_type>
            >,
            pow_expression_t<E2, 2>
        >
    > diff_type;

    constexpr div_expression(expression<E1> const& e1, expression<E2> const& e2) : e1(e1()), e2(e2()){}

    constexpr E1 const& expr1() const { return e1; }
    constexpr E2 const& expr2() const { return e2; }

    constexpr diff_type diff() const {
        return div_diff(e1, e2);
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return e1(x) / e2(x);
    }

    constexpr std::string to_string() const {
        return '(' + e1.to_string() + '/' + e2.to_string() + ')';
    }

private:
    E1 const e1;
    E2 const e2;
};

template<class E1, class E2>
constexpr div_expression<E1, E2>
operator/(expression<E1> const& e1, expression<E2> const& e2){
    return div_expression<E1, E2>(e1, e2);
}

template<class E>
constexpr int operator/(expression<E> const&, int_constant<0> const&){
    static_assert(false, "Zero division in operator/");
    return 0;
}

template<class E>
constexpr int_constant<0> const
operator/(int_constant<0> const&, expression<E> const&){
    return int_constant<0>();
}

template<class E>
constexpr E const& operator/(expression<E> const& e, int_constant<1> const&){
    return e();
}

template<class E, typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    div_expression<E, scalar<T> >
> operator/(expression<E> const& e, T const& val){
    return e / scalar<T>(val);
}

template<class E, typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    div_expression<scalar<T>, E>
> operator/(T const& val, expression<E> const& e){
    return scalar<T>(val) / e;
}

template<typename VT>
constexpr scalar<VT> operator/(scalar<VT> const& e1, scalar<VT> const& e2){
    return scalar<VT>(e1.value / e2.value);
}

template<typename VT, int N>
constexpr std::enable_if_t<N != 0, scalar<VT> >
operator/(scalar<VT> const& e1, int_constant<N>const&){
    return scalar<VT>(e1.value / N);
}

template<int N, typename VT>
constexpr std::enable_if_t<N != 0, scalar<VT> >
operator/(int_constant<N>const&, scalar<VT> const& e2){
    return scalar<VT>(N / e2.value);
}

_SYMDIFF1_END
