#pragma once

#include "scalar.hpp"

_SYMDIFF_BEGIN

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
    template<unsigned M>
    struct diff_type
    {
        typedef typename negate_expression_type<typename E::template diff_type<M>::type>::type type;
    };

    constexpr negate_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return -e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return -e(vars);
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
operator-(int_constant<N> const&){
    return int_constant<-N>();
}

template<typename VT>
constexpr scalar<VT>
operator-(scalar<VT> const& e){
    return scalar<VT>(-e.value);
}

_SYMDIFF_END
