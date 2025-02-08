#pragma once

#include "math_object.hpp"

_KIAM_MATH_BEGIN

template<typename CONT>
struct context : math_object_base<CONT>{};

template<class CB, class _Proxy = CB>
struct context_builder : math_object<CB, _Proxy>{};

#define CONTEXT_BUILDER(CB) context_builder<CB, typename CB::proxy_type>

_KIAM_MATH_END
