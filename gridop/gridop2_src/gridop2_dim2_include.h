#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

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

namespace units = math_units;

typedef units::quantity<units::scalar> nodim;
typedef units::quantity<units::length> length_type;
typedef units::quantity<units::time> time_type;
typedef units::quantity<units::mass> mass_type;
typedef units::quantity<units::velocity> velocity_type;
typedef units::quantity<units::energy> si_energy_type;

typedef typename units::divide_typeof_helper<si_energy_type, mass_type>::type energy_type;
