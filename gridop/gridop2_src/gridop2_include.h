#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

//#include <boost/math_units/systems/si.hpp>
#include <boost/throw_exception.hpp>
#include <boost/core/demangle.hpp>
#define DEMANGLE(name) boost::core::demangle(name)
#include <boost/timer/timer.hpp>

#include <kiam/math/vector_grid_function.hpp>
#include <kiam/math/binary_evaluable_objects.hpp>
#include <kiam/math/dim2_index.hpp>
#include <kiam/math/math_operator.hpp>
#include <kiam/math/kiam_math_alg.h>
using namespace _KIAM_MATH;
