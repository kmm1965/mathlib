#pragma once

#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_shifted.hpp>
#include <boost/preprocessor/enum_params.hpp>

#include "executor.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class MA, class _Proxy = MA>
struct assignment : _KIAM_MATH::math_object<MA, _Proxy>
{
    typedef TAG tag_type;
};

template<typename TAG>
struct empty_assignment : assignment<TAG, empty_assignment<TAG> >
{
    __DEVICE
    void operator()(size_t i) const {}

    template<typename CONTEXT>
    __DEVICE
    void operator()(size_t index, context<CONTEXT> const& context) const {}
};

template<typename TAG, class A0
#if MAX_ASSIGNMENT_SIZE > 1
    ,
#define PACKAGE_TPL_ARG0(n) class A##n = empty_assignment<TAG>
#define PACKAGE_TPL_ARG(z, n, unused) PACKAGE_TPL_ARG0(n)
    BOOST_PP_ENUM_SHIFTED(MAX_ASSIGNMENT_SIZE, PACKAGE_TPL_ARG, ~)
#undef PACKAGE_TPL_ARG0
#undef PACKAGE_TPL_ARG
#endif
>
struct package
{
    typedef TAG tag_type;

    package(
#define PACKAGE_PARAMS(z, n, unused) assignment<TAG, A##n, typename A##n::proxy_type> const& a##n
        BOOST_PP_ENUM(MAX_ASSIGNMENT_SIZE, PACKAGE_PARAMS, ~)
#undef PACKAGE_PARAMS
    ) :
#define PACKAGE_PROXY(z, n, unused) a##n##_proxy(a##n.get_proxy())
        BOOST_PP_ENUM(MAX_ASSIGNMENT_SIZE, PACKAGE_PROXY, ~)
#undef PACKAGE_PROXY
    {}

private:
    __DEVICE
    void invoke(size_t i) const
    {
#define PACKAGE_INVOKE(z, n, unused) a##n##_proxy(i);
        BOOST_PP_REPEAT(MAX_ASSIGNMENT_SIZE, PACKAGE_INVOKE, ~)
#undef PACKAGE_INVOKE
    }

    template<class CBP>
    __DEVICE
    void invoke(size_t i, CBP const& context_builder_proxy) const
    {
        typename CBP::context_type const context(i, context_builder_proxy);
#define PACKAGE_INVOKE(z, n, unused) a##n##_proxy(i, context);
        BOOST_PP_REPEAT(MAX_ASSIGNMENT_SIZE, PACKAGE_INVOKE, ~)
#undef PACKAGE_INVOKE
    }

public:
    template<class E>
    package& exec(size_t i_begin, size_t i_end, EXECUTOR(E) const& executor)
    {
        this->i_begin = i_begin;
        this->i_end = i_end;
        executor()(i_end - i_begin, *this);
        return *this;
    }

    package& exec(size_t i_begin, size_t i_end)
    {
        default_executor<TAG> _executor;
        return exec(i_begin, i_end, _executor);
    }

    template<class CB, class E>
    package& exec(size_t i_begin, size_t i_end, CONTEXT_BUILDER(CB) const& context_builder, EXECUTOR(E) const& executor)
    {
        this->i_begin = i_begin;
        this->i_end = i_end;
        executor()(i_end - i_begin, context_builder, *this);
        return *this;
    }

    template<class CB>
    package& exec(size_t i_begin, size_t i_end, CONTEXT_BUILDER(CB) const& context_builder)
    {
        default_executor<TAG> executor;
        return exec(i_begin, i_end, context_builder, executor);
    }

    template<typename RANGE, class E>
    package& exec1(RANGE const& range, EXECUTOR(E) const& executor)
    {
        i_begin = 0;
        i_end = std::numeric_limits<size_t>::max();
        executor()(range, *this);
        return *this;
    }

    template<typename RANGE, class CB, class E>
    package& exec1(RANGE const& range, CONTEXT_BUILDER(CB) const& context_builder, EXECUTOR(E) const& executor)
    {
        i_begin = 0;
        i_end = std::numeric_limits<size_t>::max();
        executor()(range, context_builder, *this);
        return *this;
    }

    __DEVICE
    void operator()(size_t i) const
    {
        i += i_begin;
#ifdef __CUDACC__
        if(i < i_end)
#endif
            invoke(i);
    }

    template<class CBP>
    __DEVICE
    void operator()(size_t i, CBP const& context_builder_proxy) const
    {
        i += i_begin;
#ifdef __CUDACC__
        if(i < i_end)
#endif
            invoke(i, context_builder_proxy);
    }

private:
#define PACKAGE_VAR(z, n, unused) typename A##n::proxy_type const a##n##_proxy;
    BOOST_PP_REPEAT(MAX_ASSIGNMENT_SIZE, PACKAGE_VAR, ~)
#undef PACKAGE_VAR

    size_t i_begin, i_end;
};

#define ASSIGNMENT_PARAMS(z, n, unused) assignment<TAG, A##n, typename A##n::proxy_type> const& a##n

#define ASSIGNMENT_IMPL(z, n, unused) \
    template<typename TAG, BOOST_PP_ENUM_PARAMS(n, class A)> \
    package<TAG, BOOST_PP_ENUM_PARAMS(n, A)> \
    math_assign(BOOST_PP_ENUM(n, ASSIGNMENT_PARAMS, ~)){ \
        return package<TAG, BOOST_PP_ENUM_PARAMS(n, A)>( \
            BOOST_PP_ENUM_PARAMS(n, a) \
            BOOST_PP_COMMA_IF(BOOST_PP_SUB(MAX_ASSIGNMENT_SIZE, n)) \
            BOOST_PP_ENUM(BOOST_PP_SUB(MAX_ASSIGNMENT_SIZE, n), BOOST_PP_ENUM_print_data, (empty_assignment<TAG>()))); \
    }
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_ADD(MAX_ASSIGNMENT_SIZE, 1), ASSIGNMENT_IMPL, ~)

#undef ASSIGNMENT_IMPL
#undef ASSIGNMENT_PARAMS

_KIAM_MATH_END

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_ASSIGNMENT(name) \
    template<class A, class _Proxy = A> \
    struct name##_assignment : _KIAM_MATH::assignment<name##_tag, A, _Proxy>{}
#else
#define DECLARE_MATH_ASSIGNMENT(name) \
    template<class A, class _Proxy = A> \
    using name##_assignment = _KIAM_MATH::assignment<name##_tag, A, _Proxy>
#endif
