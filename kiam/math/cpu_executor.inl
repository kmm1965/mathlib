#pragma once

#include "type_traits.hpp"

_KIAM_MATH_BEGIN

template<typename TAG>
template<class Closure>
void cpu_executor<TAG>::operator()(size_t size, Closure const& closure) const
{
#pragma omp parallel for schedule(static)
    for(ssize_t i = 0; i < (ssize_t)size; ++i)
        closure(i);
}

_KIAM_MATH_END
