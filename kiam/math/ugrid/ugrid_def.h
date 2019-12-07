#pragma once

#include "../math_operator.hpp"
#include "../context.hpp"
#include "../assignment.hpp"

#define _UGRID_MATH_BEGIN _KIAM_MATH_BEGIN namespace ugrid {
#define _UGRID_MATH_END } _KIAM_MATH_END
#define _UGRID_MATH _KIAM_MATH::ugrid

_UGRID_MATH_BEGIN

struct ugrid_tag;

DECLARE_MATH_OPERATOR(ugrid);
DECLARE_MATH_EVALUABLE_OBJECT(ugrid);
DECLARE_MATH_CONTEXT(ugrid);
DECLARE_MATH_EXECUTOR(ugrid);
DECLARE_MATH_ASSIGNMENT(ugrid);

_UGRID_MATH_END
