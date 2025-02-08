#pragma once

#include "math_def.h"
#include "closure_callback.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class E, class _Proxy = E>
struct executor : math_object<E, _Proxy>
{
    using tag_type = TAG;
};

#define EXECUTOR(E) executor<typename E::tag_type, E, typename E::proxy_type>

#if defined(__CUDACC__)

template<typename TAG>
struct cuda_executor : executor<TAG, cuda_executor<TAG> >
{
    template<class Closure>
    void operator()(size_t size, Closure const& closure) const;

    template<class CB, class Closure>
    std::enable_if_t<std::is_same<TAG, typename CB::tag_type>::value>
    operator()(size_t size, Closure const& closure, CONTEXT_BUILDER(CB) const& context_builder) const
    {
        closure_context_callback<CB, Closure> const callback(closure, context_builder);
        (*this)(size, callback);
    }
};

template<typename TAG>
using default_executor = cuda_executor<TAG>;

#elif defined(__OPENCL__)

template<typename TAG>
struct opencl_executor : executor<TAG, opencl_executor<TAG> >
{
    opencl_executor(){}

    template<class Closure>
    void operator()(size_t size, Closure const& closure) const;

    template<class CB, class Closure>
    std::enable_if_t<std::is_same<TAG, typename CB::tag_type>::value>
    operator()(size_t size, Closure const& closure, CONTEXT_BUILDER(CB) const& context_builder) const
    {
        closure_context_callback<CB, Closure> const callback(closure, context_builder);
        (*this)(size, callback);
    }
};

template<typename TAG>
using default_executor = opencl_executor<TAG>;

#else   // __CUDACC__

template<typename TAG>
struct cpu_executor : executor<TAG, cpu_executor<TAG> >
{
    cpu_executor(){}

    template<class Closure>
    void operator()(size_t size, Closure const& closure) const;

    template<class CB, class Closure>
    std::enable_if_t<std::is_same<typename CB::tag_type, TAG>::value>
    operator()(size_t size, Closure const& closure, CONTEXT_BUILDER(CB) const& context_builder) const
    {
        closure_context_callback<CB, Closure> const callback(closure, context_builder);
        (*this)(size, callback);
    }
};

template<typename TAG>
using default_executor = cpu_executor<TAG>;

#endif  // __CUDACC

_KIAM_MATH_END

#define DECLARE_MATH_EXECUTOR(name) \
    template<class E, class _Proxy = E> \
    using name##_executor = _KIAM_MATH::executor<name##_tag, E, _Proxy>; \
    using name##_default_executor = _KIAM_MATH::default_executor<name##_tag>

#if defined(__CUDACC__)
#include "cuda_executor.inl"
#elif defined(__OPENCL__)
#include "opencl_executor.inl"
#else
#include "cpu_executor.inl"
#endif
