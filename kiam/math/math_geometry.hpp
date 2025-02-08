#pragma once

#include "math_utils.hpp" // det2, det3
#include "vector2_value.hpp"
#include "vector_value.hpp"

_KIAM_MATH_BEGIN

// ѕерпендикуляр к прямой на плоскости, заданной двумя точками, из третьей точки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector2_value<T> >::type
perpendicular(vector2_value<T> const& node1, vector2_value<T> const& node2, vector2_value<T> const& node0)
{
    vector2_value<T> const n(node2.value_y() - node1.value_y(), node1.value_x() - node2.value_x());
    return node0 - ((node0 - node1) & n) * n / (n & n);
}

// Точка, симметричная относительно заданной прямой на плоскости.
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector2_value<T> >::type
symmetry(vector2_value<T> const& node1, vector2_value<T> const& node2, vector2_value<T> const& node0){
    return perpendicular<DIM>(node1, node2, node0) * 2. - node0;
}

// Перпендикуляр к плоскости в пространстве, заданной тремя точками, из четвёртой точки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_value<T> >::type
perpendicular(vector_value<T> const& node1, vector_value<T> const& node2, vector_value<T> const& node3, vector_value<T> const& node0)
{
    vector_value<T> const n = (node2 - node1) * (node3 - node1);
    return node0 - ((node0 - node1) & n) * n / (n & n);
}

// Точка, симметричная относительно заданной плоскости в пространстве.
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_value<T> >::type
symmetry(vector_value<T> const& node1, vector_value<T> const& node2, vector_value<T> const& node3, vector_value<T> const& node0){
    return perpendicular<DIM>(node1, node2, node3, node0) * 2. - node0;
}

_KIAM_MATH_END
