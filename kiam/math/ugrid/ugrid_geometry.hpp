#pragma once

#include "../vector_type.hpp"
#include "../vector_proxy.hpp"
#include "../math_array.hpp"

#include <cassert>

_UGRID_MATH_BEGIN

// Площадь треугольника на плоскости
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_area(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2){
    return func::abs((p1 - p0) * (p2 - p0)) / 2;
}

// Площадь треугольника в пространстве
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
triangle_area(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2){
    return ((p1 - p0) * (p2 - p0)).length() / 2;
}

// Площадь четырёхугольника на плоскости
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
quadrangle_area(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2, vector2_value<T> const& p3){
    // Половина длины векторного произведения диагоналей.
    return func::abs((p2 - p0) * (p3 - p1)) / 2;
}

// Площадь четырёхугольника в пространстве
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
quadrangle_area(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2, vector_value<T> const& p3){
    // Половина длины векторного произведения диагоналей.
    return ((p2 - p0) * (p3 - p1)).length() / 2;
}

// Проверка четырёхугольника на плоскости на корректность
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2>::type
check_convex_quadrangle(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2, vector2_value<T> const& p3)
{
#ifndef NDEBUG
    using vector_t = vector2_value<T>;
    T const eps = 1e-12;
    T check;
    // Проверим, что никакие две точки попарно не совпадают.
    assert((check = (p0 - p1).length()) > eps);
    assert((check = (p0 - p2).length()) > eps);
    assert((check = (p0 - p3).length()) > eps);
    assert((check = (p1 - p2).length()) > eps);
    assert((check = (p1 - p3).length()) > eps);
    assert((check = (p2 - p3).length()) > eps);
    // Проверим, что никакие три точки не лежат на одной прямой.
    assert((check = func::abs((p0 - p1) * (p0 - p2))) > eps);
    assert((check = func::abs((p0 - p1) * (p0 - p3))) > eps);
    assert((check = func::abs((p0 - p2) * (p0 - p3))) > eps);
    assert((check = func::abs((p1 - p2) * (p1 - p3))) > eps);
    // Проверим четырёхугольник на выпуклость.
    // Векторные произведения последовательных пар векторов должны лежать по одну сторону от плоскости,
    // Найдём первое такое векторное произведение:
    T const vec0 = (p1 - p0) * (p2 - p1);
    assert((check = vec0 * ((p2 - p1) * (p3 - p2))) > 0);
    assert((check = vec0 * ((p3 - p2) * (p0 - p3))) > 0);
    assert((check = vec0 * ((p0 - p3) * (p1 - p0))) > 0);
    // Площадь четырёхугольника можно посчитать или как сумму площадей двух треугольников,
    // или как половину длины векторного произведения диагоналей.
    // Убедимся, что оба способа дают близкие результаты.
    assert((check = func::abs(quadrangle_area<DIM>(p0, p1, p2, p3) - triangle_area<DIM>(p0, p1, p3) - triangle_area<DIM>(p2, p1, p3))) < eps);
#endif
}

// Проверка четырёхугольника в пространстве на корректность
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3>::type
check_convex_quadrangle(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2, vector_value<T> const& p3)
{
#ifndef NDEBUG
    using vector_t = vector_value<T>;
    T const eps = 1e-12;
    T check;
    // Проверим, что никакие две точки попарно не совпадают.
    assert((check = (p0 - p1).length()) > eps);
    assert((check = (p0 - p2).length()) > eps);
    assert((check = (p0 - p3).length()) > eps);
    assert((check = (p1 - p2).length()) > eps);
    assert((check = (p1 - p3).length()) > eps);
    assert((check = (p2 - p3).length()) > eps);
    // Проверим, что никакие три точки не лежат на одной прямой.
    assert((check = ((p0 - p1) * (p0 - p2)).length()) > eps);
    assert((check = ((p0 - p1) * (p0 - p3)).length()) > eps);
    assert((check = ((p0 - p2) * (p0 - p3)).length()) > eps);
    assert((check = ((p1 - p2) * (p1 - p3)).length()) > eps);
    // Проверим, что все четыре точки лежат на одной плоскости.
    //assert((check = func::abs((p1 & (p2 * p3)) - (p0 & (p2 * p3)) + (p0 & (p1 * p3)) - (p0 & (p1 * p2)))) < 1e-6);
    // Проверим четырёхугольник на выпуклость.
    // Векторные произведения последовательных пар векторов должны лежать по одну сторону от плоскости,
    // Найдём первое такое векторное произведение:
    vector_t const vec0 = (p1 - p0) * (p2 - p1);
    assert((check = (vec0 & ((p2 - p1) * (p3 - p2)))) > 0);
    assert((check = (vec0 & ((p3 - p2) * (p0 - p3)))) > 0);
    assert((check = (vec0 & ((p0 - p3) * (p1 - p0)))) > 0);
    // Площадь четырёхугольника можно посчитать или как сумму площадей двух треугольников,
    // или как половину длины векторного произведения диагоналей.
    // Убедимся, что оба способа дают близкие результаты.
    T const
        quad_area = quadrangle_area<DIM>(p0, p1, p2, p3),
        tri_area = (triangle_area<DIM>(p0, p1, p3) + triangle_area<DIM>(p2, p1, p3) +
            triangle_area<DIM>(p0, p1, p2) + triangle_area<DIM>(p0, p3, p2)) / 2;
    assert((check = func::abs(quad_area - tri_area) / quad_area) < 1e-1);
#endif
}

// Обход всех симплексов (треугольников) плоской фигуры (треугольник или четырёхугольник).
template<unsigned DIM, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2>::type
for_each_simplex(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, FUNC f)
{
    assert(cell_node_count == 3 || cell_node_count == 4);
    using vector_type = vector_type_t<T, DIM>;
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    f(node0, node1, node2);
    if(cell_node_count == 4){
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
        f(node0, node3, node2);
    }
}

// Обход всех симплексов плоской фигуры с суммированием.
template<unsigned DIM, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
for_each_simplex_val(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, T init, FUNC f)
{
    using vector_type = vector_type_t<T, DIM>;
    T result = init;
    for_each_simplex<DIM, T>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&result, &f](vector_type const& node0, vector_type const& node1, vector_type const& node2)
        { result += f(node0, node1, node2); });
    return result;
}

// Обход всех симплексов (треугольников) плоской фигуры (треугольник или четырёхугольник).
template<unsigned DIM, size_t N, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2>::type
for_each_simplex(_KIAM_MATH::math_array<_KIAM_MATH::vector2_value<T>, N> const& nodes, unsigned cell_node_count, FUNC f)
{
    assert(cell_node_count <= N);
    assert(cell_node_count == 3 || cell_node_count == 4);
    _KIAM_MATH::vector2_value<T> const
        &node0 = nodes[0],
        &node1 = nodes[1],
        &node2 = nodes[2];
        f(node0, node1, node2);
    if(cell_node_count == 4){
        _KIAM_MATH::vector2_value<T> const& node3 = nodes[3];
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
        f(node0, node3, node2);
    }
}

// Обход всех симплексов плоской фигуры с суммированием.
template<unsigned DIM, size_t N, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
for_each_simplex_val(_KIAM_MATH::math_array<_KIAM_MATH::vector2_value<T>, N> const& nodes, unsigned cell_node_count, T init, FUNC f)
{
    T result = init;
    for_each_simplex<DIM>(nodes, cell_node_count,
        [&result, &f](_KIAM_MATH::vector2_value<T> const& node0, _KIAM_MATH::vector2_value<T> const& node1, _KIAM_MATH::vector2_value<T> const& node2)
        { result += f(node0, node1, node2); });
    return result;
}

// Обход всех симплексов (тетраэдров) объёмной фигуры.
template<unsigned DIM, typename  T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3>::type
for_each_simplex(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, FUNC f)
{
    using vector_type = vector_type_t<T, DIM>;
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]],
        &node3 = nodes[cell_node_index[cell_node_index_start + 3]];
    switch (cell_node_count){
    case 4: // Tetrahedron
        f(node0, node1, node2, node3);
        break;
    case 5: // Pyramid
    {
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
#endif
        vector_type const& node4 = nodes[cell_node_index[cell_node_index_start + 4]];
        f(node0, node1, node2, node4);
        f(node0, node3, node2, node4);
        break;
    }
    case 6: // Wedge
    {
        const vector_type
            &node4 = nodes[cell_node_index[cell_node_index_start + 4]],
            &node5 = nodes[cell_node_index[cell_node_index_start + 5]];
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node4, node3);
        check_convex_quadrangle<DIM>(node0, node2, node5, node3);
        check_convex_quadrangle<DIM>(node1, node2, node5, node4);
#endif
        f(node0, node1, node2, node3);
        f(node1, node2, node3, node5);
        f(node1, node3, node4, node5);
        break;
    }
    case 8: // Brick
    {
        const vector_type
            &node4 = nodes[cell_node_index[cell_node_index_start + 4]],
            &node5 = nodes[cell_node_index[cell_node_index_start + 5]],
            &node6 = nodes[cell_node_index[cell_node_index_start + 6]],
            &node7 = nodes[cell_node_index[cell_node_index_start + 7]];
#if 0 //ndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
        check_convex_quadrangle<DIM>(node4, node5, node6, node7);
        check_convex_quadrangle<DIM>(node0, node1, node5, node4);
        check_convex_quadrangle<DIM>(node2, node3, node7, node6);
        check_convex_quadrangle<DIM>(node0, node3, node7, node4);
        check_convex_quadrangle<DIM>(node1, node2, node6, node5);
#endif
        f(node0, node2, node3, node6);
        f(node0, node2, node5, node6);
        f(node0, node1, node2, node5);
        f(node0, node4, node5, node6);
        f(node0, node3, node4, node6);
        f(node3, node4, node6, node7);
        break;
    }
    default:
        assert(false);
#ifndef __CUDACC__
        throw std::runtime_error("Invalid number of nodes");
#endif
    }
}

// Обход всех симплексов (тетраэдров) объёмной фигуры с суммированием.
template<unsigned DIM, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
for_each_simplex_val(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, T init, FUNC f)
{
    using vector_type = vector_type_t<T, DIM>;
    T result = init;
    for_each_simplex<DIM, T>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&result, &f](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3)
        { result += f(node0, node1, node2, node3); });
    return result;
}

// Обход всех симплексов (тетраэдров) объёмной фигуры.
template<unsigned DIM, size_t N, typename  T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3>::type
for_each_simplex(math_array<vector_value<T>, N> const& nodes, unsigned cell_node_count, FUNC f)
{
    assert(cell_node_count <= N);
    using vector_type = vector_value<T>;
    const vector_type
        &node0 = nodes[0],
        &node1 = nodes[1],
        &node2 = nodes[2],
        &node3 = nodes[3];
    switch (cell_node_count){
    case 4: // Tetrahedron
        f(node0, node1, node2, node3);
        break;
    case 5: // Pyramid
    {
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
#endif
        vector_type const& node4 = nodes[4];
        f(node0, node1, node2, node4);
        f(node0, node3, node2, node4);
        break;
    }
    case 6: // Wedge
    {
        const vector_type
            &node4 = nodes[4],
            &node5 = nodes[5];
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node4, node3);
        check_convex_quadrangle<DIM>(node0, node2, node5, node3);
        check_convex_quadrangle<DIM>(node1, node2, node5, node4);
#endif
        f(node0, node1, node2, node3);
        f(node1, node2, node3, node5);
        f(node1, node3, node4, node5);
        break;
    }
    case 8: // Brick
    {
        const vector_type
            &node4 = nodes[4],
            &node5 = nodes[5],
            &node6 = nodes[6],
            &node7 = nodes[7];
#if 0 //ndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
        check_convex_quadrangle<DIM>(node4, node5, node6, node7);
        check_convex_quadrangle<DIM>(node0, node1, node5, node4);
        check_convex_quadrangle<DIM>(node2, node3, node7, node6);
        check_convex_quadrangle<DIM>(node0, node3, node7, node4);
        check_convex_quadrangle<DIM>(node1, node2, node6, node5);
#endif
        f(node0, node2, node3, node6);
        f(node0, node2, node5, node6);
        f(node0, node1, node2, node5);
        f(node0, node4, node5, node6);
        f(node0, node3, node4, node6);
        f(node3, node4, node6, node7);
        break;
    }
    default:
        assert(false);
#ifndef __CUDACC__
        throw std::runtime_error("Invalid number of nodes");
#endif
    }
}

// Обход всех симплексов (тетраэдров) объёмной фигуры с суммированием.
template<unsigned DIM, size_t N, typename T, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
for_each_simplex_val(math_array<vector2_value<T>, N> const& nodes, unsigned cell_node_count, T init, FUNC f)
{
    using vector_type = vector_type_t<T, DIM>;
    T result = init;
    for_each_simplex<DIM>(nodes, cell_node_count,
        [&result, &f](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3)
        { result += f(node0, node1, node2, node3); });
    return result;
}

// Объём тетраэдра
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_volume(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    return func::abs(((node1 - node0) * (node2 - node0)) & (node3 - node0)) / 6;
}

// Площадь плоской ячейки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
cell_volume(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using vector_type = vector2_value<T>;
    assert(cell_node_count == 3 || cell_node_count == 4);
    vector_type const
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    if(cell_node_count == 3)
        return triangle_area<DIM>(node0, node1, node2);
    else {
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
#endif
        return quadrangle_area<DIM>(node0, node1, node2, node3);
    }
}

// Объём пространственной ячейки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
cell_volume(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using vector_type = vector_value<T>;
    return for_each_simplex_val<DIM, T>(nodes, cell_node_index, cell_node_index_start, cell_node_count, 0,
        [](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3){
            return tetrahedron_volume<DIM>(node0, node1, node2, node3);
        });
}

// Центр масс четырёхугольника
template<unsigned DIM, typename T>
__HOST __DEVICE
vector_type_t<T, DIM> quadrilateral_mcenter(const vector_type_t<T, DIM> &node0, const vector_type_t<T, DIM> &node1,
    const vector_type_t<T, DIM> &node2, const vector_type_t<T, DIM> &node3)
{
    check_convex_quadrangle<DIM>(node0, node1, node2, node3);
    using vector_type = vector_type_t<T, DIM>;
    // Разбиваем четырёхугольник на два треугольника,
    // находим центры масс этих треугольников и их площади (массы),
    // а затем находим центр масс двух материальных точек с этими массами в этих центрах.
    const vector_type
        p1 = (node0 + node1 + node2) / 3.,
        p2 = (node0 + node3 + node2) / 3.;
    T const
        S1 = triangle_area<DIM>(node0, node1, node2),
        S2 = triangle_area<DIM>(node0, node3, node2),
        S12 = S1 + S2;
    vector_type const mcenter = p1 + (p2 - p1) * (S2 / (S1 + S2));
    vector_type const
        p_1 = (node0 + node1 + node3) / 3.,
        p_2 = (node2 + node1 + node3) / 3.;
    T const
        S_1 = triangle_area<DIM>(node0, node1, node3),
        S_2 = triangle_area<DIM>(node2, node1, node3),
        S_12 = S_1 + S_2;
    const vector_type mcenter1 = p_1 + (p_2 - p_1) * (S_2 / (S_1 + S_2));
    return mcenter + (mcenter1 - mcenter) * (S_12 / (S12 + S_12));
}

// Центр масс плоской ячейки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector_type_t<T, DIM> >::type
cell_mcenter(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    assert(cell_node_count == 3 || cell_node_count == 4);
    using vector_type = vector_type_t<T, DIM>;
    vector_type const
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    return cell_node_count == 3 ? (node0 + node1 + node2) / 3. :
        quadrilateral_mcenter<DIM, T>(node0, node1, node2, nodes[cell_node_index[cell_node_index_start + 3]]);
}

template<unsigned DIM, size_t N, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector_type_t<T, DIM> >::type
cell_mcenter(math_array<vector2_value<T>, N> const& nodes, unsigned cell_node_count)
{
    assert(cell_node_count == 3 || cell_node_count == 4);
    vector2_value<T> const
        &node0 = nodes[0],
        &node1 = nodes[1],
        &node2 = nodes[2];
    return cell_node_count == 3 ? (node0 + node1 + node2) / 3. :
        quadrilateral_mcenter<DIM, T>(node0, node1, node2, nodes[3]);
}

// Центр масс пространственной ячейки
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_type_t<T, DIM> >::type
cell_mcenter(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using vector_type = vector_type_t<T, DIM>;
    vector_type mcenter;
    T V = 0;
    for_each_simplex<DIM, T>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&mcenter, &V](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3)
        {
            vector_type const mcenter1 = (node0 + node1 + node2 + node3) / 4.;
            T const V1 = tetrahedron_volume<DIM>(node0, node1, node2, node3);
            mcenter += (mcenter1 - mcenter) * (V1 / (V += V1));
        });
    return mcenter;
}

template<unsigned DIM, size_t N, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_type_t<T, DIM> >::type
cell_mcenter(math_array<vector_value<T>, N> const& nodes, unsigned cell_node_count)
{
    using vector_type = vector_value<T>;
    vector_type mcenter;
    T V = 0;
    for_each_simplex<DIM>(nodes, cell_node_count,
        [&mcenter, &V](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3)
        {
            vector_type const mcenter1 = (node0 + node1 + node2 + node3) / 4.;
            T const V1 = tetrahedron_volume<DIM>(node0, node1, node2, node3);
            mcenter += (mcenter1 - mcenter) * (V1 / (V += V1));
        });
    return mcenter;
}

// Длина грани на плоскости
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
calc_interface_area(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 2);
    return (nodes[interface_node_index[interface_node_index_start + 1]] - nodes[interface_node_index[interface_node_index_start]]).length();
}

// Площадь грани в пространстве
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
calc_interface_area(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 3 || interface_node_count == 4);
    using vector_type = vector_type_t<T, DIM>;
    const vector_type
        &node0 = nodes[interface_node_index[interface_node_index_start]],
        &node1 = nodes[interface_node_index[interface_node_index_start + 1]],
        &node2 = nodes[interface_node_index[interface_node_index_start + 2]];
    if(interface_node_count == 3)
        return triangle_area<DIM>(node0, node1, node2);
    else {
        vector_type const& node3 = nodes[interface_node_index[interface_node_index_start + 3]];
#ifndef NDEBUG
        check_convex_quadrangle<DIM>(node0, node1, node2, node3);
#endif
        return quadrangle_area<DIM>(node0, node1, node2, node3);
    }
}

// Минимальная высота треугольника на плоскости.
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_min_height(vector2_value<T> const& node0, vector2_value<T> const& node1, vector2_value<T> const& node2)
{
    T const
        volume = std::abs((node0 - node1) * (node0 - node2)),
        max_length = func::max(func::max((node0 - node1).length(), (node1 - node2).length()), (node2 - node0).length());
    assert(volume > 0);
    return volume / max_length;
}

// Высота треугольника на плоскости, опущенная из узла node0.
template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_height(vector2_value<T> const& node0, vector2_value<T> const& node1, vector2_value<T> const& node2)
{
    T const
        volume = std::abs((node0 - node1) * (node0 - node2)),
        length = (node1 - node2).length();
    assert(volume > 0);
    return volume / length;
}

// Высота ячейки на плоскости.
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
calc_cell_height(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using vector_type = vector2_value<T>;
    assert(cell_node_count >= 3);
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    switch (cell_node_count){
    case 3:
        return triangle_min_height<DIM>(node0, node1, node2);
    case 4:
    {
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
        // Минимальную высоту четырёхугольника определим как минимум всех высот.
        return func::min(func::min(func::min(func::min(func::min(func::min(func::min(
            triangle_height<T, DIM>(node0, node1, node2),
            triangle_height<T, DIM>(node0, node2, node3)),
            triangle_height<T, DIM>(node1, node0, node2)),
            triangle_height<T, DIM>(node1, node2, node3)),
            triangle_height<T, DIM>(node2, node0, node1)),
            triangle_height<T, DIM>(node2, node1, node3)),
            triangle_height<T, DIM>(node3, node0, node1)),
            triangle_height<T, DIM>(node3, node1, node2));
    }
    default:
        assert(false);
        return T();
    }
}

// Минимальная высота тетраэдра
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_min_height(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    T const
        volume = std::abs(((node0 - node1) * (node0 - node2)) & (node0 - node3)),
        max_area = func::max(func::max(func::max(
        ((node0 - node1) * (node0 - node2)).length(),
            ((node1 - node2) * (node1 - node3)).length()),
            ((node2 - node0) * (node2 - node3)).length()),
            ((node3 - node0) * (node3 - node1)).length());
    assert(volume > 0);
    assert(max_area > 0);
    return volume / max_area;
}

// Высота тетраэдра, опущенная из узла node0.
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_height(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    T const
        volume = std::abs(((node0 - node1) * (node0 - node2)) & (node0 - node3)),
        area = ((node1 - node2) * (node1 - node3)).length();
    assert(volume > 0);
    assert(area > 0);
    return volume / area;
}

// Высота ячейки в пространстве
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
calc_cell_height(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using vector_type = vector_type_t<T, DIM>;
    T height = std::numeric_limits<T>::max();
    for_each_simplex<DIM, T>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&height](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3){
            height = func::min(height, tetrahedron_min_height<DIM>(node0, node1, node2, node3));
        });
    return height;
}

// Нормаль к грани на плоскости
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector2_value<T> >::type
calc_normal(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 2);
    vector2_value<T> const vec = nodes[interface_node_index[interface_node_index_start + 1]] - nodes[interface_node_index[interface_node_index_start]];
    return vector2_value<T>(vec.value_y(), -vec.value_x());
}

// Нормаль к грани в пространстве
template<unsigned DIM, typename T>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_value<T> >::type
calc_normal(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 3 || interface_node_count == 4);
    vector_value<T> const
        &node0 = nodes[interface_node_index[interface_node_index_start]],
        vec1 = nodes[interface_node_index[interface_node_index_start + 2]] - node0,
        vec2 = interface_node_count == 3 ? nodes[interface_node_index[interface_node_index_start + 1]] - node0 :
            nodes[interface_node_index[interface_node_index_start + 3]] - nodes[interface_node_index[interface_node_index_start + 1]];
    return vec1 * vec2;
}

_UGRID_MATH_END
