#pragma once

#include "symdiff1_def.h"
#include "../math_object.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct expression : kiam::math::math_object_base<E>
{
protected: // protect from direct construction
    constexpr expression() {}
};

template<class E>
constexpr std::ostream &operator<<(std::ostream &o, const expression<E> &e) {
    return o << e().to_string();
}

_SYMDIFF1_END
