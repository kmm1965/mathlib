#pragma once

#include "scalar.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct negate_expression;

template<class E>
struct negate_expression_type
{
    typedef typename std::conditional<
        is_int_constant<E>::value,
        int_constant<-int_constant_value<E>::value>,
        std::conditional_t<is_scalar<E>::value, E, negate_expression<E> >
    >::type type;
};

template<class E>
using negate_expression_t = typename negate_expression_type<E>::type;

template<class E>
struct negate_expression : expression<negate_expression<E> >
{
    typedef negate_expression_t<typename E::diff_type> diff_type;

    constexpr negate_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return -e.diff();
    }

    template<typename T>
    constexpr T operator()(const T &x) const {
        return -e(x);
    }

    constexpr std::string to_string() const {
        return "(-" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr negate_expression<E>
operator-(const expression<E>& e){
    return negate_expression<E>(e);
}

template<int N>
constexpr int_constant<-N>
operator-(const int_constant<N>&){
    return int_constant<-N>();
}

template<typename VT>
constexpr scalar<VT>
operator-(const scalar<VT> &e){
    return scalar<VT>(-e.value);
}

_SYMDIFF1_END
