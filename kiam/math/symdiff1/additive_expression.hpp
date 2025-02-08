#pragma once

#include "int_constant.hpp"
#include "negate_expression.hpp"

_SYMDIFF1_BEGIN

template<typename T>
struct is_plus : std::false_type {};

template<typename T>
struct is_plus<std::plus<T> > : std::true_type {};

template<typename T>
bool constexpr is_plus_v = is_plus<T>::value;

template<class E1, typename OP, class E2>
struct additive_expression;

template<class E1, typename OP, class E2>
struct additive_expression_type
{
    typedef std::conditional_t<
        is_int_constant_v<E1> && is_int_constant_v<E2>,
        std::conditional_t<is_plus_v<OP>,
            int_constant<int_constant_val<E1> + int_constant_val<E2> >,
            int_constant<int_constant_val<E1> - int_constant_val<E2> >
        >,
        std::conditional_t<std::is_same_v<E2, int_constant<0> >, E1,
            std::conditional_t<std::is_same_v<E1, int_constant<0> >,
                std::conditional_t<is_plus_v<OP>, E2, negate_expression_t<E2> >,
                std::conditional_t<
                    is_scalar_v<E1> && (std::is_same_v<E1, E2> || is_int_constant_v<E2>), E1,
                    std::conditional_t<
                        is_int_constant_v<E1> && is_scalar_v<E2>, E2,
                        additive_expression<E1, OP, E2>
                    >
                >
            >
        >
    > type;
};

template<class E1, typename OP, class E2>
using additive_expression_t = typename additive_expression_type<E1, OP, E2>::type;

template<typename OP, class E1, class E2>
constexpr std::enable_if_t<is_plus_v<OP>,
    additive_expression_t<typename E1::diff_type, OP, typename E2::diff_type>
> additive_expression_diff(E1 const& e1, E2 const& e2){
    return e1.diff() + e2.diff();
}

template<typename OP, class E1, class E2>
constexpr std::enable_if_t<!is_plus_v<OP>,
    additive_expression_t<typename E1::diff_type, OP, typename E2::diff_type>
> additive_expression_diff(const E1& e1, const E2& e2){
    return e1.diff() - e2.diff();
}

template<class E1, typename OP, class E2>
struct additive_expression : expression<additive_expression<E1, OP, E2> >{
    typedef additive_expression_t<typename E1::diff_type, OP, typename E2::diff_type> diff_type;
    
    constexpr additive_expression(expression<E1> const& e1, OP const& op, expression<E2> const& e2) : e1(e1()), op(op), e2(e2()){}

    constexpr E1 const& expr1() const { return e1; }
    constexpr E2 const& expr2() const { return e2; }

    constexpr diff_type diff() const {
        return additive_expression_diff<OP>(e1, e2);
    }

    template<typename T>
    T operator()(T const& x) const {
        return op(e1(x), e2(x));
    }

    constexpr std::string to_string() const {
        return '(' + e1.to_string() + (is_plus_v<OP> ? '+' : '-') + e2.to_string() + ')';
    }

private:
    E1 const e1;
    OP const op;
    E2 const e2;
};

template<class E1, class E2>
constexpr additive_expression<E1, std::plus<double>, E2>
operator+(expression<E1> const& e1, expression<E2> const& e2){
    return additive_expression<E1, std::plus<double>, E2>(e1, std::plus<double>(), e2);
}

template<class E1, class E2>
constexpr additive_expression<E1, std::minus<double>, E2>
operator-(expression<E1> const& e1, expression<E2> const& e2){
    return additive_expression<E1, std::minus<double>, E2>(e1, std::minus<double>(), e2);
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
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    additive_expression<E, std::plus<double>, scalar<T> >
> operator+(expression<E> const& e, T const& val){
    return e + scalar<T>(val);
}

template<class E, typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    additive_expression<scalar<T>, std::plus<double>, E>
> operator+(T const& val, expression<E> const& e){
    return scalar<T>(val) + e;
}

template<class E, typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    additive_expression<E, std::minus<double>, scalar<T> >
> operator-(expression<E> const& e, T const& val){
    return e - scalar<T>(val);
}

template<class E, typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value,
    additive_expression<scalar<T>, std::minus<double>, E>
> operator-(T const& val, expression<E> const& e){
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
operator+(scalar<VT> const& e1, int_constant<N>const&){
    return scalar<VT>(e1.value + N);
}

template<typename VT, int N>
constexpr scalar<VT>
operator-(scalar<VT> const& e1, int_constant<N>const&){
    return scalar<VT>(e1.value - N);
}

template<int N, typename VT>
constexpr scalar<VT>
operator+(int_constant<N>const&, scalar<VT> const& e2){
    return scalar<VT>(N + e2.value);
}

template<int N, typename VT>
constexpr scalar<VT>
operator-(int_constant<N>const&, scalar<VT> const& e2){
    return scalar<VT>(N - e2.value);
}

_SYMDIFF1_END
