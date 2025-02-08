#pragma once

#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>

#include "context.hpp"
#include "grid_expression.hpp"

#ifndef MAX_MATH_OPERATOR_PARAMS
#define MAX_MATH_OPERATOR_PARAMS 2
#endif

_KIAM_MATH_BEGIN

template<class MO, class _Proxy = MO>
struct math_operator;

#define MATH_OP(MO) _KIAM_MATH::math_operator<MO, typename MO::proxy_type>

template<class MO, class GEXP>
struct operator_evaluator : grid_expression<typename MO::template get_tag_type<typename GEXP::tag_type>::type, operator_evaluator<MO, GEXP> >
{
    typedef typename MO::template get_tag_type<typename GEXP::tag_type>::type tag_type;
    typedef typename MO::template get_value_type<get_value_type_t<GEXP> >::type value_type;

    operator_evaluator(MATH_OP(MO) const& op, GRID_EXPR(GEXP) const& gexp) : op_proxy(op.get_proxy()), gexp_proxy(gexp.get_proxy()){}

    IMPLEMENT_DEFAULT_COPY_CONSRUCTOR(operator_evaluator);

    __DEVICE
    value_type operator[](size_t i) const {
        return op_proxy(i, gexp_proxy);
    }

    __DEVICE
    value_type operator()(size_t i) const {
        return (*this)[i];
    }

    template<typename CONTEXT>
    __DEVICE
    value_type operator()(size_t i, context<CONTEXT> const& ctx) const {
        return op_proxy(i, ctx(), gexp_proxy);
    }

//private:
    typename MO::proxy_type const op_proxy;
    typename GEXP::proxy_type const gexp_proxy;
};

#define MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE(z, n, unused) typename GEXP##n::tag_type
#define MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE(z, n, unused) get_value_type_t<GEXP##n>
#define MATH_OPERATOR_EVALUATOR_GEXP_TYPE(z, n, unused) typedef GRID_EXPR(GEXP##n) gexp##n##_type;
#define MATH_OPERATOR_EVALUATOR_GEXP(z, n, unused) gexp##n##_type const& gexp##n
#define MATH_OPERATOR_EVALUATOR_GEXP_PROXY_INIT(z, n, unused) gexp##n##_proxy(gexp##n.get_proxy())
#define MATH_OPERATOR_EVALUATOR_GEXP_PROXY(z, n, unused) gexp##n##_proxy
#define MATH_OPERATOR_EVALUATOR_GEXP_PROXY_DEF(z, n, unused) typename GEXP##n::proxy_type const gexp##n##_proxy;

#define MATH_OPERATOR_EVALUATOR(z, n, unused) \
    template<class MO, BOOST_PP_ENUM_PARAMS(n, class GEXP)> \
    struct operator_evaluator##n : grid_expression<typename MO::template get_tag_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE, ~)>::type, operator_evaluator##n<MO, BOOST_PP_ENUM_PARAMS(n, GEXP)> > \
    { \
        typedef typename MO::template get_tag_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE, ~)>::type tag_type; \
        typedef typename MO::template get_value_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE, ~)>::type value_type; \
        BOOST_PP_REPEAT(n, MATH_OPERATOR_EVALUATOR_GEXP_TYPE, ~) \
        operator_evaluator##n(MATH_OP(MO) const& op, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_GEXP, ~)) : \
            op_proxy(op.get_proxy()), BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_GEXP_PROXY_INIT, ~){} \
        __DEVICE \
        value_type operator[](size_t i) const { \
            return op_proxy(i, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_GEXP_PROXY, ~)); \
        } \
        __DEVICE \
        value_type operator()(size_t i) const { \
            return op_proxy(i, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_GEXP_PROXY, ~)); \
        } \
        template<typename CONTEXT> \
        __DEVICE \
        value_type operator()(size_t i, context<CONTEXT> const& ctx) const { \
            return op_proxy(i, ctx, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_GEXP_PROXY, ~)); \
        } \
    private: \
        typename MO::proxy_type const op_proxy; \
        BOOST_PP_REPEAT(n, MATH_OPERATOR_EVALUATOR_GEXP_PROXY_DEF, ~) \
    };

BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), MATH_OPERATOR_EVALUATOR, ~)
#undef MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE
#undef MATH_OPERATOR_EVALUATOR_GEXP_TYPE
#undef MATH_OPERATOR_EVALUATOR_GEXP
#undef MATH_OPERATOR_EVALUATOR_GEXP_PROXY_INIT
#undef MATH_OPERATOR_EVALUATOR_GEXP_PROXY
#undef MATH_OPERATOR_EVALUATOR_GEXP_PROXY_DEF
#undef MATH_OPERATOR_EVALUATOR

template<class MO, class _Proxy>
struct math_operator : math_object<MO, _Proxy>
{
    template<typename EO_TAG>
    struct get_tag_type
    {
        typedef EO_TAG type;
    };

    template<typename T>
    struct get_value_type
    {
        typedef T type;
    };

#define MATH_OPERATOR_TAG(z, n, unused) TAG
#define MATH_OPERATOR_GET_TAG_TYPE(z, n, unused) \
    template<BOOST_PP_ENUM_PARAMS(n, typename TAG)> struct get_tag_type##n; \
    template<typename TAG> struct get_tag_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_TAG, ~)>{ \
        typedef TAG type; \
    };
    BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), MATH_OPERATOR_GET_TAG_TYPE, ~)
#undef MATH_OPERATOR_TAG
#undef MATH_OPERATOR_GET_TAG_TYPE

#define MATH_OPERATOR_T(z, n, unused) T
#define MATH_OPERATOR_GET_VALUE_TYPE(z, n, unused) \
    template<BOOST_PP_ENUM_PARAMS(n, typename T)> struct get_value_type##n; \
    template<typename T> struct get_value_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_T, ~)>{ \
        typedef T type; \
    };
    BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), MATH_OPERATOR_GET_VALUE_TYPE, ~)
#undef MATH_OPERATOR_T
#undef MATH_OPERATOR_GET_VALUE_TYPE

protected:
    math_operator(){} // Protect from direct construction
};

_KIAM_MATH_END

#define IMPLEMENT_MATH_EVAL_OPERATOR(type) \
    template<class GEXP> \
    _KIAM_MATH::operator_evaluator<type, GEXP> operator()(GRID_EXPR(GEXP) const& gexp) const { \
        return operator_evaluator<type, GEXP>(*this, gexp); \
    }

#define IMPLEMENT_MATH_EVAL_OPERATOR_GEXP(z, n, unused) GRID_EXPR(GEXP##n) const& gexp##n

#define IMPLEMENT_MATH_EVAL_OPERATOR_N_I(n, type) \
    template<BOOST_PP_ENUM_PARAMS(n, class GEXP)> \
    _KIAM_MATH::operator_evaluator##n<type, BOOST_PP_ENUM_PARAMS(n, GEXP)> \
    operator()(BOOST_PP_ENUM(n, IMPLEMENT_MATH_EVAL_OPERATOR_GEXP, ~)) const { \
        return _KIAM_MATH::operator_evaluator##n<type, BOOST_PP_ENUM_PARAMS(n, GEXP)>(*this, BOOST_PP_ENUM_PARAMS(n, gexp)); \
    }

#define IMPLEMENT_MATH_EVAL_OPERATOR_N(n, type) IMPLEMENT_MATH_EVAL_OPERATOR_N_I(n, type)

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_OPERATOR(name) \
    template<class MO, class GEXP> \
    struct name##_operator_evaluator : _KIAM_MATH::operator_evaluator<MO, GEXP>{}; \
    template<class MO, class _Proxy = MO> \
    struct name##_operator : _KIAM_MATH::math_operator<MO, _Proxy>{}
#else
#define DECLARE_MATH_OPERATOR(name) \
    template<class MO, class GEXP> \
    using name##_operator_evaluator = _KIAM_MATH::operator_evaluator<MO, GEXP>; \
    template<class MO, class _Proxy = MO> \
    using name##_operator = _KIAM_MATH::math_operator<MO, _Proxy>
#endif

_KIAM_MATH_BEGIN

template<typename T, typename F>
struct inplace_math_operator : math_operator<inplace_math_operator<T, F> >
{
    template<typename T1>
    struct get_value_type {
        typedef T type;
    };

    inplace_math_operator(F f) : f(f){}

    template<class GEXP_P>
    __DEVICE
    T operator()(size_t i, GEXP_P const& gexp_proxy) const {
        return f(i, gexp_proxy);
    }

    template<typename CONTEXT, class GEXP_P>
    __DEVICE
    T operator()(size_t i, context<CONTEXT> const& ctx, GEXP_P const& gexp_proxy) const {
        return f(i, ctx(), gexp_proxy);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(inplace_math_operator)

private:
    F const f;
};

template<typename T, typename F>
inplace_math_operator<T, F> get_inplace_math_operator(F f){
    return inplace_math_operator<T, F>(f);
}

template<typename T, typename F>
struct inplace_math_operator2 : math_operator<inplace_math_operator2<T, F> >
{
    template<typename T1, typename T2>
    struct get_value_type2 {
        typedef T type;
    };

    inplace_math_operator2(F f) : f(f){}

    template<class EOP0, class GEXP1_P>
    __DEVICE
    T operator()(size_t i, EOP0 const& gexp0_proxy, GEXP1_P const& gexp1_proxy) const {
        return f(i, gexp0_proxy, gexp1_proxy);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR_N(2, inplace_math_operator2)

private:
    F const f;
};

template<typename T, typename F>
inplace_math_operator2<T, F> get_inplace_math_operator2(F f){
    return inplace_math_operator2<T, F>(f);
}

template<typename F>
struct inplace_apply_operator : math_operator<inplace_apply_operator<F> >
{
    inplace_apply_operator(F f) : f(f){}

    template<class GEXP_P>
    __DEVICE
    void operator()(size_t i, GEXP_P const& gexp_proxy) const {
        f(i, gexp_proxy);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(inplace_apply_operator)

private:
    F const f;
};

template<typename F>
inplace_apply_operator<F> get_inplace_apply_operator(F f){
    return inplace_apply_operator<F>(f);
}

#define INPLACE_APPLY_OPERATOR_EOP(z, n, unused) GEXP_P##n &gexp##n##_proxy
#define INPLACE_APPLY_OPERATOR_EOP_PROXY(z, n, unused) gexp##n##_proxy

#define INPLACE_APPLY_OPERATOR(z, n, unused) \
    template<typename F> \
    struct inplace_apply_operator##n : math_operator<inplace_apply_operator##n<F> > { \
        inplace_apply_operator##n(F f) : f(f){} \
        template<BOOST_PP_ENUM_PARAMS(n, class GEXP_P)> \
        __DEVICE void operator()(size_t i, BOOST_PP_ENUM(n, INPLACE_APPLY_OPERATOR_EOP, ~)) const { \
            f(i, BOOST_PP_ENUM(n, INPLACE_APPLY_OPERATOR_EOP_PROXY, ~)); \
        } \
        IMPLEMENT_MATH_EVAL_OPERATOR_N(n, inplace_apply_operator##n) \
    private: F const f; \
    }; \
    template<typename F> \
    inplace_apply_operator##n<F> get_inplace_apply_operator##n(F f){ \
        return inplace_apply_operator##n<F>(f); \
    }

BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), INPLACE_APPLY_OPERATOR, ~)

#undef INPLACE_APPLY_OPERATOR_EOP
#undef INPLACE_APPLY_OPERATOR_EOP_PROXY

_KIAM_MATH_END
