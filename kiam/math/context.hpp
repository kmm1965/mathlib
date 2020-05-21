#pragma once

#include "math_object.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename CONTEXT>
struct context : math_object_base<CONTEXT>{};

template<typename TAG, class CB, class _Proxy = CB>
struct context_builder : math_object<CB, _Proxy>{};

template<typename TAG, class CBP>
struct context_builder_proxy : math_object_base<CBP>{};

_KIAM_MATH_END

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_CONTEXT(name) \
    template<typename CONTEXT> \
    struct name##_context : _KIAM_MATH::context<name##_tag, CONTEXT>{}; \
    template<class CB, class _Proxy = CB> \
    struct name##_context_builder : _KIAM_MATH::context_builder<name##_tag, CB, _Proxy>{}; \
    template<class CBP> \
    struct name##_context_builder_proxy : _KIAM_MATH::context_builder_proxy<name##_tag, CBP>{}
#else
#define DECLARE_MATH_CONTEXT(name) \
    template<typename CONTEXT> \
    using name##_context = _KIAM_MATH::context<name##_tag, CONTEXT>; \
    template<class CB, class _Proxy = CB> \
    using name##_context_builder = _KIAM_MATH::context_builder<name##_tag, CB, _Proxy>; \
    template<class CBP> \
    using name##_context_builder_proxy = _KIAM_MATH::context_builder_proxy<name##_tag, CBP>
#endif
