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

#include <boost/units/systems/si.hpp>
namespace units = boost::units;
namespace si = units::si;

typedef units::quantity<si::dimensionless> nodim;
typedef units::quantity<si::length> length_type;
typedef units::quantity<si::time> time_type;
typedef units::quantity<si::mass> mass_type;
typedef units::quantity<si::velocity> velocity_type;
typedef units::quantity<si::energy> si_energy_type;

typedef typename units::divide_typeof_helper<si_energy_type, mass_type>::type energy_type;

typedef units::quantity<si::mass_density> density_type;
typedef units::quantity<
    typename units::divide_typeof_helper<si::energy, si::volume>::type
> specific_energy_type;

#include <kiam/math/vector_grid_function.hpp>
#include <kiam/math/binary_evaluable_objects.hpp>
#include <kiam/math/negate_evaluable_object.hpp>
#include <kiam/math/index_base.hpp>
#include <kiam/math/math_operator.hpp>
#include <kiam/math/kiam_math_alg.h>
#include <kiam/math/boost_units_support.hpp>
using namespace _KIAM_MATH;
