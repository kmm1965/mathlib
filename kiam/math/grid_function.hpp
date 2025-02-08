#pragma once

#include "grid_expression.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class GF, class _Proxy = GF>
struct grid_function : grid_expression<TAG, GF, _Proxy>{};

_KIAM_MATH_END
