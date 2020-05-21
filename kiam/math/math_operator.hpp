#pragma once

#include "context.hpp"
#include "evaluable_object.hpp"
#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>

#ifndef MAX_MATH_OPERATOR_PARAMS
#define MAX_MATH_OPERATOR_PARAMS 2
#endif

_KIAM_MATH_BEGIN

template<class MO, class _Proxy = MO>
struct math_operator;

#define MATH_OP(MO) _KIAM_MATH::math_operator<MO, typename MO::proxy_type>

template<class MO, class EO>
struct operator_evaluator : evaluable_object<typename MO::template get_tag_type<typename EO::tag_type>::type, operator_evaluator<MO, EO> >
{
    typedef typename MO::template get_tag_type<typename EO::tag_type>::type tag_type;
    typedef typename MO::template get_value_type<get_value_type_t<EO> >::type value_type;

    CONSTEXPR operator_evaluator(MATH_OP(MO) const& op, EOBJ(EO) const& eobj) : op_proxy(op.get_proxy()), eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    value_type operator[](size_t i) const {
        return op_proxy(i, eobj_proxy);
    }

    __DEVICE
    value_type operator()(size_t i) const {
        return op_proxy(i, eobj_proxy);
    }

    template<typename CONTEXT>
    __DEVICE
    value_type operator()(size_t i, context<tag_type, CONTEXT> const& context) const {
        return op_proxy(i, eobj_proxy, context);
    }

private:
    const typename MO::proxy_type op_proxy;
    const typename EO::proxy_type eobj_proxy;
};

#define MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE(z, n, unused) typename EO##n::tag_type
#define MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE(z, n, unused) get_value_type_t<EO##n>
#define MATH_OPERATOR_EVALUATOR_EOBJ_TYPE(z, n, unused) typedef EOBJ(EO##n) eobj##n##_type;
#define MATH_OPERATOR_EVALUATOR_EOBJ(z, n, unused) eobj##n##_type const& eobj##n
#define MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_INIT(z, n, unused) eobj##n##_proxy(eobj##n.get_proxy())
#define MATH_OPERATOR_EVALUATOR_EOBJ_PROXY(z, n, unused) eobj##n##_proxy
#define MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_DEF(z, n, unused) const typename EO##n::proxy_type eobj##n##_proxy;

#define MATH_OPERATOR_EVALUATOR(z, n, unused) \
    template<class MO, BOOST_PP_ENUM_PARAMS(n, class EO)> \
    struct operator_evaluator##n : evaluable_object<typename MO::template get_tag_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE, ~)>::type, operator_evaluator##n<MO, BOOST_PP_ENUM_PARAMS(n, EO)> > \
    { \
        typedef typename MO::template get_tag_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_TAG_TYPE, ~)>::type tag_type; \
        typedef typename MO::template get_value_type##n<BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE, ~)>::type value_type; \
        BOOST_PP_REPEAT(n, MATH_OPERATOR_EVALUATOR_EOBJ_TYPE, ~) \
        operator_evaluator##n(MATH_OP(MO) const& op, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EOBJ, ~)) : \
            op_proxy(op.get_proxy()), BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_INIT, ~){} \
        __DEVICE \
        value_type operator[](size_t i) const { \
            return op_proxy(i, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EOBJ_PROXY, ~)); \
        } \
        __DEVICE \
        value_type operator()(size_t i) const { \
            return op_proxy(i, BOOST_PP_ENUM(n, MATH_OPERATOR_EVALUATOR_EOBJ_PROXY, ~)); \
        } \
    private: \
        const typename MO::proxy_type op_proxy; \
        BOOST_PP_REPEAT(n, MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_DEF, ~) \
    };

BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), MATH_OPERATOR_EVALUATOR, ~)
#undef MATH_OPERATOR_EVALUATOR_EO_VALUE_TYPE
#undef MATH_OPERATOR_EVALUATOR_EOBJ_TYPE
#undef MATH_OPERATOR_EVALUATOR_EOBJ
#undef MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_INIT
#undef MATH_OPERATOR_EVALUATOR_EOBJ_PROXY
#undef MATH_OPERATOR_EVALUATOR_EOBJ_PROXY_DEF
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

protected: // protect from direct construction
    math_operator(){}
};

_KIAM_MATH_END

#define REIMPLEMENT_OPERATOR_EXEC() \
    template<typename EOP> \
    __DEVICE \
    typename super::template get_value_type<_KIAM_MATH::get_value_type_t<EOP> >::type operator()(size_t i, EOP const& eobj_proxy) const { \
        return super::operator()(i, eobj_proxy); \
    }

#define IMPLEMENT_MATH_EVAL_OPERATOR(type) \
    template<class EO> \
    _KIAM_MATH::operator_evaluator<type, EO> operator()(EOBJ(EO) const& eobj) const { \
        return operator_evaluator<type, EO>(*this, eobj); \
    }

#define IMPLEMENT_MATH_EVAL_OPERATOR_EOBJ(z, n, unused) EOBJ(EO##n) const& eobj##n

#define IMPLEMENT_MATH_EVAL_OPERATOR_N(n, type) \
    template<BOOST_PP_ENUM_PARAMS(n, class EO)> \
    _KIAM_MATH::operator_evaluator##n<type, BOOST_PP_ENUM_PARAMS(n, EO)> \
    operator()(BOOST_PP_ENUM(n, IMPLEMENT_MATH_EVAL_OPERATOR_EOBJ, ~)) const { \
        return _KIAM_MATH::operator_evaluator##n<type, BOOST_PP_ENUM_PARAMS(n, EO)>(*this, BOOST_PP_ENUM_PARAMS(n, eobj)); \
    }

#define IMPLEMENT_MATH_EVAL_OPERATOR2(type) IMPLEMENT_MATH_EVAL_OPERATOR_N(2, type)
#if MAX_MATH_OPERATOR_PARAMS > 2
#define IMPLEMENT_MATH_EVAL_OPERATOR3(type) IMPLEMENT_MATH_EVAL_OPERATOR_N(3, type)
#if MAX_MATH_OPERATOR_PARAMS > 3
#define IMPLEMENT_MATH_EVAL_OPERATOR4(type) IMPLEMENT_MATH_EVAL_OPERATOR_N(4, type)
#if MAX_MATH_OPERATOR_PARAMS > 4
#define IMPLEMENT_MATH_EVAL_OPERATOR5(type) IMPLEMENT_MATH_EVAL_OPERATOR_N(5, type)
#endif
#endif
#endif

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_OPERATOR(name) \
    template<class MO, class EO> \
    struct name##_operator_evaluator : _KIAM_MATH::operator_evaluator<MO, EO>{}; \
    template<class MO, class _Proxy = MO> \
    struct name##_operator : _KIAM_MATH::math_operator<MO, _Proxy>{}
#else
#define DECLARE_MATH_OPERATOR(name) \
    template<class MO, class EO> \
    using name##_operator_evaluator = _KIAM_MATH::operator_evaluator<MO, EO>; \
    template<class MO, class _Proxy = MO> \
    using name##_operator = _KIAM_MATH::math_operator<MO, _Proxy>
#endif

_KIAM_MATH_BEGIN

template<typename F>
struct inplace_math_operator : math_operator<inplace_math_operator<F> >
{
    inplace_math_operator(F f) : f(f) {}

    template<typename EOP>
    double operator()(int i, int j, EOP const& eobj_proxy) const {
        return f(i, j, eobj_proxy);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(inplace_math_operator)

private:
    F f;
};

template<typename F>
inplace_math_operator<F> get_inplace_math_operator(F f){
    return inplace_math_operator<F>(f);
}

_KIAM_MATH_END
