#pragma once

#include "funcprog2_setup.h"

_FUNCPROG2_BEGIN

template<typename FuncType, typename FuncImpl>
struct function2;

template<typename Ret, typename... Args, typename FuncImpl>
struct function2<Ret(Args...), FuncImpl>
{
    using result_type = Ret;

    __DEVICE __HOST constexpr function2(FuncImpl const& impl) : impl(impl){}

    __DEVICE result_type operator()(Args... args) const {
        return impl(args...);
    }

private:
    FuncImpl const impl;
};

#define FUNCTION2_(FuncType, FuncImpl) BOOST_IDENTITY_TYPE((function2<FuncType, FuncImpl>))
#define FUNCTION2(FuncType, FuncImpl) typename FUNCTION2_(FuncType, FuncImpl)

_FUNCPROG2_END
