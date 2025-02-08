#pragma once

#include "int_constant.hpp"
#include "negate_expression.hpp"

_SYMDIFF_BEGIN

template<class E1, char op, class E2>
struct additive_expression;

template<class E1, char op, class E2>
struct additive_expression_type
{
    typedef std::conditional_t<std::is_same_v<E2, int_constant<0> >, E1,
        std::conditional_t<std::is_same_v<E1, int_constant<0> >,
            std::conditional_t<op == '+', E2, negate_expression_t<E2> >,
            std::conditional_t<is_int_constant_v<E1> && is_int_constant_v<E2>,
                std::conditional_t<op == '+',
                    int_constant<int_constant_val<E1> + int_constant_val<E2> >,
                    int_constant<int_constant_val<E1> - int_constant_val<E2> >
                >,
                std::conditional_t<
                    is_scalar_v<E1> && (std::is_same_v<E1, E2> || is_int_constant_v<E2>), E1,
                    std::conditional_t<is_int_constant_v<E1> && is_scalar_v<E2>, E2,
                        additive_expression<E1, op, E2>
                    >
                >
            >
        >
    > type;
};

template<class E1, char op, class E2>
using additive_expression_t = typename additive_expression_type<E1, op, E2>::type;

template<class E1, char op, class E2>
struct additive_expression : expression<additive_expression<E1, op, E2> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef additive_expression_t<
            typename E1::template diff_type<M>::type,
            op,
            typename E2::template diff_type<M>::type
        > type;
    };
    
    constexpr additive_expression(expression<E1> const& e1, expression<E2> const& e2) : e1(e1()), e2(e2()){}

    constexpr E1 const& expr1() const { return e1; }
    constexpr E2 const& expr2() const { return e2; }

    template<unsigned M>
    constexpr typename std::enable_if<op == '+', typename diff_type<M>::type>::type
    diff() const {
        return e1.diff<M>() + e2.diff<M>();
    }

    template<unsigned M>
    constexpr typename std::enable_if<op == '-', typename diff_type<M>::type>::type
    diff() const {
        return e1.diff<M>() - e2.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr typename std::enable_if<op == '+', T>::type
    operator()(std::array<T, _Size> const& vars) const {
        return e1(vars) + e2(vars);
    }

    template<typename T, size_t _Size>
    constexpr typename std::enable_if<op == '-', T>::type
    operator()(std::array<T, _Size> const& vars) const {
        return e1(vars) - e2(vars);
    }

private:
    E1 const e1;
    E2 const e2;
};

template<class E1, class E2>
constexpr additive_expression<E1, '+', E2>
operator+(expression<E1> const& e1, expression<E2> const& e2){
    return additive_expression<E1, '+', E2>(e1, e2);
}

template<class E1, class E2>
constexpr additive_expression<E1, '-', E2>
operator-(expression<E1> const& e1, expression<E2> const& e2){
    return additive_expression<E1, '-', E2>(e1, e2);
}

template<int N1, int N2>
constexpr int_constant<N1 + N2>
operator+(int_constant<N1> const&, int_constant<N2> const&){
    return int_constant<N1 + N2>();
}

template<int N1, int N2>
constexpr int_constant<N1 - N2>
operator-(int_constant<N1> const&, int_constant<N2> const&){
    return int_constant<N1 - N2>();
}

template<class E>
constexpr E const& operator+(expression<E> const& e, int_constant<0> const&){
    return e();
}

template<class E>
constexpr E const& operator-(expression<E> const& e, int_constant<0> const&){
    return e();
}

template<class E>
constexpr E const& operator+(int_constant<0> const&, expression<E> const& e){
    return e();
}

template<class E>
constexpr negate_expression_t<E>
operator-(int_constant<0> const&, expression<E> const& e){
    return -e;
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    additive_expression<E, '+', scalar<T> >
>::type operator+(expression<E> const& e, T const& val){
    return e + scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    additive_expression<scalar<T>, '+', E>
>::type operator+(T const& val, expression<E> const& e){
    return scalar<T>(val) + e;
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    additive_expression<E, '-', scalar<T> >
>::type operator-(expression<E> const& e, T const& val){
    return e - scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
    additive_expression<scalar<T>, '-', E>
>::type operator-(T const& val, expression<E> const& e){
    return scalar<T>(val) - e;
}

template<typename VT>
constexpr scalar<VT>
operator+(scalar<VT> const& e1, scalar<VT> const& e2){
    return scalar<VT>(e1.value + e2.value);
}

template<typename VT>
constexpr scalar<VT>
operator-(scalar<VT> const& e1, scalar<VT> const& e2){
    return scalar<VT>(e1.value - e2.value);
}

template<typename VT, int N>
constexpr scalar<VT>
operator+(scalar<VT> const& e1, int_constant<N> const&){
    return scalar<VT>(e1.value + N);
}

template<typename VT, int N>
constexpr scalar<VT>
operator-(scalar<VT> const& e1, int_constant<N> const&){
    return scalar<VT>(e1.value - N);
}

template<int N, typename VT>
constexpr scalar<VT>
operator+(int_constant<N> const&, scalar<VT> const& e2){
    return scalar<VT>(N + e2.value);
}

template<int N, typename VT>
constexpr scalar<VT>
operator-(int_constant<N> const&, scalar<VT> const& e2){
    return scalar<VT>(N - e2.value);
}

_SYMDIFF_END
