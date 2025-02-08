#pragma once

#include <list>
#include <string>

#ifdef __CUDACC__
#define __DEVICE __device__
#define __HOST __host__
#else
#define __DEVICE
#define __HOST
#endif

#include "../math_def.h"

#define _FUNCPROG2_BEGIN _KIAM_MATH_BEGIN namespace funcprog2 {
#define _FUNCPROG2_END } _KIAM_MATH_END

#define _FUNCPROG2 _KIAM_MATH::funcprog2

_FUNCPROG2_BEGIN

template<typename T>
using list_t = std::list<T>;

template<typename A>
struct EmptyData {};

using None = EmptyData<void>;

template<typename F>
using base_class_t = typename F::base_class;

_FUNCPROG2_END
