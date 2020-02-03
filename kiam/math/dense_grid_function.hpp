#pragma once

#include "vector_grid_function.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
using dense_grid_function = vector_grid_function<TAG, T>;

_KIAM_MATH_END

#define DECLARE_MATH_DENSE_GRID_FUNCTION(name) \
    template<typename T> \
    using name##_dense_grid_function = _KIAM_MATH::dense_grid_function<name##_tag, T>
