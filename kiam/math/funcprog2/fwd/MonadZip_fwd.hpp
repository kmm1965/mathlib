#pragma once

#include "../funcprog2_common.hpp"

_FUNCPROG2_BEGIN

template<typename T>
struct MonadZip;

template<typename T>
using MonadZip_t = MonadZip<base_class_t<T> >;

_FUNCPROG2_END
