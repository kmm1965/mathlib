#pragma once

#include "context.hpp"

_KIAM_MATH_BEGIN

#ifndef __CUDACC__
template<typename TAG>
struct _evaluable_object
{
    typedef TAG tag_type;
    using base_class = _evaluable_object;
};
#endif

template<typename TAG, class EO, class _Proxy = EO>
struct evaluable_object : math_object<EO, _Proxy>
#ifndef __CUDACC__
    , _evaluable_object<TAG>
#endif
{
    typedef TAG tag_type;

    template<typename T>
    __DEVICE
    void assign(T &val, size_t i) const {
        val = (*this)()[i];
    }

    template<typename T, typename CONTEXT>
    __DEVICE
    void assign(T &val, size_t i, const context<tag_type, CONTEXT>& context) const {
        val = (*this)()(i, context);
    }

protected: // protect from direct construction
    CONSTEXPR evaluable_object() {}
};

_KIAM_MATH_END

#define EOBJ(EO) _KIAM_MATH::evaluable_object<typename EO::tag_type, EO, typename EO::proxy_type>

#include "func_evaluable_object.hpp"

#ifndef __CUDACC__
#include "evaluable_object_func.hpp"
#endif

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_EVALUABLE_OBJECT(name) \
    template<class EO, class _Proxy = EO> \
    struct name##_evaluable_object : _KIAM_MATH::evaluable_object<name##_tag, EO, _Proxy>{}
#else
#define DECLARE_MATH_EVALUABLE_OBJECT(name) \
    template<class EO, class _Proxy = EO> \
    using name##_evaluable_object = _KIAM_MATH::evaluable_object<name##_tag, EO, _Proxy>
#endif
