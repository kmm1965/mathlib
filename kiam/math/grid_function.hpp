#pragma once

#include "evaluable_object.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class GF, class _Proxy = GF>
struct grid_function : evaluable_object<TAG, GF, _Proxy>{};

_KIAM_MATH_END
