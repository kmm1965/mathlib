#pragma once

#include "../math_def.h"

#define _SYMDIFF_BEGIN _KIAM_MATH_BEGIN namespace symdiff {
#define _SYMDIFF_END } _KIAM_MATH_END
#define _SYMDIFF _KIAM_MATH::symdiff

#if MATH_USE_CPP17
#  include <any>
#  define ANY_CAST(type, any_value) std::any_cast<type>(any_value)
  _SYMDIFF_BEGIN using sd_any = std::any; _SYMDIFF_END
#else
#  include <boost/any.hpp>
#  define ANY_CAST(type, any_value) boost::any_cast<type>(any_value)
  _SYMDIFF_BEGIN using sd_any = boost::any; _SYMDIFF_END
#endif // MATH_USE_CPP17
