#pragma once

#include "vector_value.hpp"
#include "vector2_value.hpp"

_KIAM_MATH_BEGIN

template<typename T, unsigned DIM>
struct vector_type;

template<typename T, unsigned DIM>
using vector_type_t = typename vector_type<T, DIM>::type;

template<typename T>
struct vector_type<T, 3>
{
    typedef vector_value<T> type;
};

template<typename T>
struct vector_type<T, 2>
{
    typedef vector2_value<T> type;
};

_KIAM_MATH_END
