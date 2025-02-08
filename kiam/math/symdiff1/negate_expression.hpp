#pragma once

#include "scalar.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct negate_expression;

template<class E>
struct negate_expression_type
{
    typedef std::conditional_t<
        is_int_constant_v<E>,
        int_constant<-int_constant_val<E> >,
        std::conditional_t<is_scalar_v<E>, E, negate_expression<E> >
    > type;
};

template<class E>
using negate_expression_t = typename negate_expression_type<E>::type;

template<class E>
struct negate_expression : expression<negate_expression<E> >
{
    typedef negate_expression_t<typename E::diff_type> diff_type;

    constexpr negate_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return -e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return -e(x);
    }

    constexpr std::string to_string() const {
        return "(-" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr negate_expression<E>
operator-(expression<E> const& e){
    return negate_expression<E>(e);
}

template<int N>
constexpr int_constant<-N>
operator-(int_constant<N>const&){
    return int_constant<-N>();
}

template<typename VT>
constexpr scalar<VT>
operator-(scalar<VT> const& e){
    return scalar<VT>(-e.value);
}

_SYMDIFF1_END
