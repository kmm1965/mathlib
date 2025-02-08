#pragma once

#include "additive_expression.hpp"

_SYMDIFF_BEGIN

template<class E1, class E2>
struct mul_expression;

template<class E1, class E2>
struct mul_expression_type
{
    typedef std::conditional_t<
        std::is_same_v<E1, int_constant<0> > || std::is_same_v<E2, int_constant<0> >,
        int_constant<0>,
        std::conditional_t<std::is_same_v<E1, int_constant<1> >, E2,
            std::conditional_t<std::is_same_v<E2, int_constant<1> >, E1,
                std::conditional_t<
                    is_int_constant_v<E1> && is_int_constant_v<E2>,
                    int_constant<int_constant_val<E1> * int_constant_val<E2> >,
                    std::conditional_t<
                        is_scalar_v<E1> && (std::is_same_v<E1, E2> || is_int_constant_v<E2>), E1,
                        std::conditional_t<
                            is_int_constant_v<E1> && is_scalar_v<E2>, E2,
                            mul_expression<E1, E2>
                        >
                    >
                >
            >
        >
    > type;
};

template<class E1, class E2>
using mul_expression_t = typename mul_expression_type<E1, E2>::type;

template<class E1, class E2>
struct mul_expression : expression<mul_expression<E1, E2> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef additive_expression_t<
            mul_expression_t<typename E1::template diff_type<M>::type, E2>,
            '+',
            mul_expression_t<E1, typename E2::template diff_type<M>::type>
        > type;
    };

    constexpr mul_expression(expression<E1> const& e1, expression<E2> const& e2) : e1(e1()), e2(e2()){}

    constexpr E1 const& expr1() const { return e1; }
    constexpr E2 const& expr2() const { return e2; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return e1.diff<M>() * e2 + e1 * e2.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return e1(vars) * e2(vars);
    }

private:
    E1 const e1;
    E2 const e2;
};

template<class E1, class E2>
constexpr mul_expression<E1, E2>
operator*(expression<E1> const& e1, expression<E2> const& e2){
    return mul_expression<E1, E2>(e1, e2);
}

template<int N1, int N2>
constexpr int_constant<N1 * N2>
operator*(int_constant<N1> const&, int_constant<N2> const&){
    return int_constant<N1 * N2>();
}

template<class E>
constexpr int_constant<0>
operator*(expression<E> const&, int_constant<0> const&){
    return int_constant<0>();
}

template<class E>
constexpr int_constant<0>
operator*(int_constant<0> const&, expression<E> const&){
    return int_constant<0>();
}

template<class E>
constexpr E const& operator*(expression<E> const& e, int_constant<1> const&){
    return e();
}

template<class E>
constexpr E const& operator*(int_constant<1> const&, expression<E> const& e){
    return e();
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    mul_expression<E, scalar<T> >
>::type operator*(expression<E> const& e, T const& val){
    return e * scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    mul_expression<scalar<T>, E>
>::type operator*(T const& val, expression<E> const& e){
    return scalar<T>(val) * e;
}

template<typename VT>
constexpr scalar<VT>
operator*(scalar<VT> const& e1, scalar<VT> const& e2){
    return scalar<VT>(e1.value * e2.value);
}

template<typename VT, int N>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator*(scalar<VT> const& e1, int_constant<N> const&){
    return scalar<VT>(e1.value * N);
}

template<int N, typename VT>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator*(int_constant<N> const&, scalar<VT> const& e2){
    return scalar<VT>(N * e2.value);
}

_SYMDIFF_END
