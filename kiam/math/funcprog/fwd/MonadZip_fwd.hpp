#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<typename T>
struct MonadZip;

template<typename T>
using MonadZip_t = MonadZip<base_class_t<T> >;

_FUNCPROG_END
