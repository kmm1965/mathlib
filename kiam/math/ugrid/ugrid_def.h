#pragma once

#include "ugrid_def0.h"

#include "../math_operator.hpp"
#include "../context.hpp"
#include "../assignment.hpp"
#include "../vector_grid_function.hpp"

_UGRID_MATH_BEGIN

struct ugrid_tag;

DECLARE_MATH_OPERATOR(ugrid);
DECLARE_MATH_GRID_EXPRESSION(ugrid);
DECLARE_MATH_EXECUTOR(ugrid);
DECLARE_MATH_ASSIGNMENT(ugrid);
DECLARE_MATH_VECTOR_GRID_FUNCTION(ugrid);

_UGRID_MATH_END
