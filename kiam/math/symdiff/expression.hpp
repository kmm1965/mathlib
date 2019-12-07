#pragma once

#include "symdiff_def.h"
#include "../math_object.hpp"

_SYMDIFF_BEGIN

template<class E>
struct expression : kiam::math::math_object_base<E>
{
protected: // protect from direct construction
    constexpr expression() {}
};

_SYMDIFF_END
