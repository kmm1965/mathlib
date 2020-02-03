#pragma once

#include "math_def.h"
#include "closure_callback.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class E, class _Proxy = E>
struct executor : math_object<E, _Proxy>
{
    typedef TAG tag_type;
};

#if defined(__CUDACC__)

template<typename TAG>
struct cuda_math_executor : executor<TAG, cuda_math_executor<TAG> >
{
    template<class Closure>
    void operator()(Closure &closure, size_t size){
        cuda_exec_callback(closure, size);
    }

    template<class Closure, class CB>
    void operator()(Closure &closure, size_t size, const context_builder<TAG, CB, typename CB::proxy_type> &context_builder){
        cuda_exec_callback(closure, size, context_builder);
    }
};

#ifdef DONT_USE_CXX_11
template<typename TAG>
struct default_executor : cuda_math_executor<TAG>{};
#else
template<typename TAG>
using default_executor = cuda_math_executor<TAG>;
#endif

#elif defined(__OPENCL__)

template<typename TAG>
struct opencl_math_executor : executor<TAG, opencl_math_executor<TAG> >
{
    template<class Closure>
    void operator()(Closure &closure, size_t size) {
        opencl_exec_callback(closure, size);
    }

    template<class Closure, class CB>
    void operator()(Closure &closure, size_t size, const context_builder<TAG, CB, typename CB::proxy_type> &context_builder) {
        opencl_exec_callback(closure, size, context_builder);
    }
};

#ifdef DONT_USE_CXX_11
template<typename TAG>
struct default_executor : opencl_math_executor<TAG> {};
#else
template<typename TAG>
using default_executor = opencl_math_executor<TAG>;
#endif

#else   // __CUDACC

template<class Callback>
void serial_exec_callback(Callback& callback, size_t size);

template<typename TAG>
struct serial_executor : executor<TAG, serial_executor<TAG> >
{
    template<class Closure>
    void operator()(Closure &closure, size_t size){
        serial_exec_callback(closure, size);
    }

    template<class Closure, class CB>
    void operator()(Closure &closure, size_t size, const context_builder<TAG, CB, typename CB::proxy_type> &context_builder)
    {
        closure_context_callback<TAG, Closure, CB> callback(closure, context_builder);
        serial_exec_callback(callback, size);
    }
};

#ifdef DONT_USE_CXX_11
template<typename TAG>
struct default_executor : serial_executor<TAG>{};
#else
template<typename TAG>
using default_executor = serial_executor<TAG>;
#endif

#endif  // __CUDACC

_KIAM_MATH_END

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_EXECUTOR(name) \
    template<class E, class _Proxy = E> \
    struct name##_executor : _KIAM_MATH::executor<name##_tag, E, _Proxy>{}; \
    typedef _KIAM_MATH::default_executor<name##_tag> default_##name##_executor
#else
#define DECLARE_MATH_EXECUTOR(name) \
    template<class E, class _Proxy = E> \
    using name##_executor = _KIAM_MATH::executor<name##_tag, E, _Proxy>; \
    typedef _KIAM_MATH::default_executor<name##_tag> name##_default_executor
#endif

#if defined(__CUDACC__)
#include "cuda_executor.inl"
#elif defined(__OPENCL__)
#include "opencl_executor.inl"
#else
#include "executor.inl"
#endif
