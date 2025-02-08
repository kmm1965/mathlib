#pragma once

#include "context.hpp"

_KIAM_MATH_BEGIN

template<class CB, class Closure>
struct closure_context_callback
{
    closure_context_callback(Closure &closure_, CONTEXT_BUILDER(CB) const& context_builder) :
        closure(closure_), context_builder_proxy(context_builder.get_proxy()){}

    __DEVICE
    void operator()(size_t i){
        closure(i, context_builder_proxy);
    }

private:
    Closure closure;
    typename CB::proxy_type context_builder_proxy;
};

_KIAM_MATH_END
