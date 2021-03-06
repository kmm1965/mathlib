#pragma once

#include "additive_expression.hpp"

_SYMDIFF_BEGIN

template<class E1, class E2>
struct mul_expression;

template<class E1, class E2>
struct mul_expression_type
{
    typedef typename std::conditional<
        std::is_same<E1, int_constant<0> >::value || std::is_same<E2, int_constant<0> >::value,
        int_constant<0>,
        typename std::conditional<std::is_same<E1, int_constant<1> >::value, E2,
            typename std::conditional<std::is_same<E2, int_constant<1> >::value, E1,
                typename std::conditional<
                    is_int_constant<E1>::value && is_int_constant<E2>::value,
                    int_constant<int_constant_value<E1>::value * int_constant_value<E2>::value>,
                    typename std::conditional<
                        is_scalar<E1>::value && (std::is_same<E1, E2>::value || is_int_constant<E2> ::value), E1,
                        typename std::conditional<
                            is_int_constant<E1>::value && is_scalar<E2>::value, E2,
                            mul_expression<E1, E2>
                        >::type
                    >::type
                >::type
            >::type
        >::type
    >::type type;
};

template<class E1, class E2>
struct mul_expression : expression<mul_expression<E1, E2> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef typename additive_expression_type<
            typename mul_expression_type<typename E1::template diff_type<M>::type, E2>::type,
            '+',
            typename mul_expression_type<E1, typename E2::template diff_type<M>::type>::type
        >::type type;
    };

    constexpr mul_expression(const expression<E1> &e1, const expression<E2> &e2) : e1(e1()), e2(e2()){}

    constexpr const E1& expr1() const { return e1; }
    constexpr const E2& expr2() const { return e2; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return e1.diff<M>() * e2 + e1 * e2.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return e1(vars) * e2(vars);
    }

private:
    const E1 e1;
    const E2 e2;
};

template<class E1, class E2>
constexpr mul_expression<E1, E2>
operator*(const expression<E1> &e1, const expression<E2> &e2){
    return mul_expression<E1, E2>(e1, e2);
}

template<int N1, int N2>
constexpr int_constant<N1 * N2>
operator*(const int_constant<N1>&, const int_constant<N2>&){
    return int_constant<N1 * N2>();
}

template<class E>
constexpr const int_constant<0>
operator*(const expression<E>&, const int_constant<0>&){
    return int_constant<0>();
}

template<class E>
constexpr const int_constant<0>
operator*(const int_constant<0>&, const expression<E>&){
    return int_constant<0>();
}

template<class E>
constexpr const E& operator*(const expression<E> &e, const int_constant<1>&){
    return e();
}

template<class E>
constexpr const E& operator*(const int_constant<1>&, const expression<E> &e){
    return e();
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    mul_expression<E, scalar<T> >
>::type operator*(const expression<E> &e, const T &val){
    return e * scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    mul_expression<scalar<T>, E>
>::type operator*(const T &val, const expression<E> &e){
    return scalar<T>(val) * e;
}

template<typename VT>
constexpr scalar<VT>
operator*(const scalar<VT> &e1, const scalar<VT> &e2){
    return scalar<VT>(e1.value * e2.value);
}

template<typename VT, int N>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator*(const scalar<VT> &e1, const int_constant<N>&){
    return scalar<VT>(e1.value * N);
}

template<int N, typename VT>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator*(const int_constant<N>&, const scalar<VT> &e2){
    return scalar<VT>(N * e2.value);
}

_SYMDIFF_END
