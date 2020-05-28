#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<class Callback>
void serial_exec_callback(Callback& callback, size_t size)
{
#pragma omp parallel for schedule(static)
    for (isize_t i = 0; i < (isize_t)size; ++i)
        callback[i];
}

_KIAM_MATH_END
