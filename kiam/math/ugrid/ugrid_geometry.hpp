#pragma once

#include "../vector_proxy.hpp"

_UGRID_MATH_BEGIN

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_area(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2){
    return func::abs((p1 - p0) * (p2 - p0)) / 2;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
triangle_area(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2){
    return ((p1 - p0) * (p2 - p0)).length() / 2;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
quadrangle_area(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2, vector2_value<T> const& p3){
    // �������� ����� ���������� ������������ ����������.
    return func::abs((p2 - p0) * (p3 - p1)) / 2;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
quadrangle_area(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2, vector_value<T> const& p3){
    // �������� ����� ���������� ������������ ����������.
    return ((p2 - p0) * (p3 - p1)).length() / 2;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2>::type
check_convex_quadrangle(vector2_value<T> const& p0, vector2_value<T> const& p1, vector2_value<T> const& p2, vector2_value<T> const& p3)
{
#ifndef NDEBUG
    using value_type = T;
    using vector_t = vector2_value<T>;
    value_type const eps = 1e-12;
    value_type check;
    // ��������, ��� ������� ��� ����� ������� �� ���������.
    assert((check = (p0 - p1).length()) > eps);
    assert((check = (p0 - p2).length()) > eps);
    assert((check = (p0 - p3).length()) > eps);
    assert((check = (p1 - p2).length()) > eps);
    assert((check = (p1 - p3).length()) > eps);
    assert((check = (p2 - p3).length()) > eps);
    // ��������, ��� ������� ��� ����� �� ����� �� ����� ������.
    assert((check = func::abs((p0 - p1) * (p0 - p2))) > eps);
    assert((check = func::abs((p0 - p1) * (p0 - p3))) > eps);
    assert((check = func::abs((p0 - p2) * (p0 - p3))) > eps);
    assert((check = func::abs((p1 - p2) * (p1 - p3))) > eps);
    // �������� �������������� �� ����������.
    // ��������� ������������ ���������������� ��� �������� ������ ������ �� ���� ������� �� ���������,
    // ����� ������ ����� ��������� ������������:
    value_type const vec0 = (p1 - p0) * (p2 - p1);
    assert(vec0 * ((p2 - p1) * (p3 - p2)) > 0);
    assert(vec0 * ((p3 - p2) * (p0 - p3)) > 0);
    assert(vec0 * ((p0 - p3) * (p1 - p0)) > 0);
    // ������� ��������������� ����� ��������� ��� ��� ����� �������� ���� �������������,
    // ��� ��� �������� ����� ���������� ������������ ����������.
    // ��������, ��� ��� ������� ���� ������� ����������.
    assert((check = func::abs(quadrangle_area<T, DIM>(p0, p1, p2, p3) - triangle_area<T, DIM>(p0, p1, p3) - triangle_area<T, DIM>(p2, p1, p3))) < eps);
#endif
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3>::type
check_convex_quadrangle(vector_value<T> const& p0, vector_value<T> const& p1, vector_value<T> const& p2, vector_value<T> const& p3)
{
#ifndef NDEBUG
    using value_type = T;
    using vector_t = vector_value<T>;
    value_type const eps = 1e-12;
    value_type check;
    // ��������, ��� ������� ��� ����� ������� �� ���������.
    assert((check = (p0 - p1).length()) > eps);
    assert((check = (p0 - p2).length()) > eps);
    assert((check = (p0 - p3).length()) > eps);
    assert((check = (p1 - p2).length()) > eps);
    assert((check = (p1 - p3).length()) > eps);
    assert((check = (p2 - p3).length()) > eps);
    // ��������, ��� ������� ��� ����� �� ����� �� ����� ������.
    assert((check = ((p0 - p1) * (p0 - p2)).length()) > eps);
    assert((check = ((p0 - p1) * (p0 - p3)).length()) > eps);
    assert((check = ((p0 - p2) * (p0 - p3)).length()) > eps);
    assert((check = ((p1 - p2) * (p1 - p3)).length()) > eps);
    // ��������, ��� ��� ������ ����� ����� �� ����� ���������.
    //assert((check = func::abs((p1 & (p2 * p3)) - (p0 & (p2 * p3)) + (p0 & (p1 * p3)) - (p0 & (p1 * p2)))) < 1e-6);
    // �������� �������������� �� ����������.
    // ��������� ������������ ���������������� ��� �������� ������ ������ �� ���� ������� �� ���������,
    // ����� ������ ����� ��������� ������������:
    vector_t const vec0 = (p1 - p0) * (p2 - p1);
    assert((vec0 & ((p2 - p1) * (p3 - p2))) > 0);
    assert((vec0 & ((p3 - p2) * (p0 - p3))) > 0);
    assert((vec0 & ((p0 - p3) * (p1 - p0))) > 0);
    // ������� ��������������� ����� ��������� ��� ��� ����� �������� ���� �������������,
    // ��� ��� �������� ����� ���������� ������������ ����������.
    // ��������, ��� ��� ������� ���� ������� ����������.
    const value_type
        quad_area = quadrangle_area<T, DIM>(p0, p1, p2, p3),
        tri_area = (triangle_area<T, DIM>(p0, p1, p3) + triangle_area<T, DIM>(p2, p1, p3) +
            triangle_area<T, DIM>(p0, p1, p2) + triangle_area<T, DIM>(p0, p3, p2)) / 2;
    //assert((check = func::abs(quad_area - tri_area) / quad_area) < 1e-2);
#endif
}

template<typename T, unsigned DIM, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2>::type
for_each_simplex(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, FUNC f)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    assert(cell_node_count == 3 || cell_node_count == 4);
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    f(node0, node1, node2);
    if (cell_node_count == 4) {
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
        check_convex_quadrangle<value_type, DIM>(node0, node1, node2, node3);
        f(node0, node3, node2);
    }
}

template<typename T, unsigned DIM, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
for_each_simplex_val(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, T init, FUNC f)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    value_type result = init;
    for_each_simplex<T, DIM>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&result, &f](vector_type const& node0, vector_type const& node1, vector_type const& node2) {
            result += f(node0, node1, node2); });
    return result;
}

template<typename  T, unsigned DIM, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3>::type
for_each_simplex(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, FUNC f)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]],
        &node3 = nodes[cell_node_index[cell_node_index_start + 3]];
    switch (cell_node_count) {
    case 4: // Tetrahedron
        f(node0, node1, node2, node3);
        break;
    case 5: // Pyramid
    {
#ifndef NDEBUG
        check_convex_quadrangle<value_type, DIM>(node0, node1, node2, node3);
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
        check_convex_quadrangle<value_type, DIM>(node0, node1, node4, node3);
        check_convex_quadrangle<value_type, DIM>(node0, node2, node5, node3);
        check_convex_quadrangle<value_type, DIM>(node1, node2, node5, node4);
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
#ifndef NDEBUG
        check_convex_quadrangle<value_type, DIM>(node0, node1, node2, node3);
        check_convex_quadrangle<value_type, DIM>(node4, node5, node6, node7);
        check_convex_quadrangle<value_type, DIM>(node0, node1, node5, node4);
        check_convex_quadrangle<value_type, DIM>(node2, node3, node7, node6);
        check_convex_quadrangle<value_type, DIM>(node0, node3, node7, node4);
        check_convex_quadrangle<value_type, DIM>(node1, node2, node6, node5);
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

template<typename T, unsigned DIM, typename FUNC>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
for_each_simplex_val(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count, T init, FUNC f)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    value_type result = init;
    for_each_simplex<T, DIM>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&result, &f](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3){
            result += f(node0, node1, node2, node3); });
    return result;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_volume(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    return func::abs(((node1 - node0) * (node2 - node0)) & (node3 - node0)) / 6;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
cell_volume(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using value_type = T;
    using vector_type = vector2_value<value_type>;
    assert(cell_node_count == 3 || cell_node_count == 4);
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    if (cell_node_count == 3)
        return triangle_area<T, DIM>(node0, node1, node2);
    else {
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
#ifndef NDEBUG
        check_convex_quadrangle<T, DIM>(node0, node1, node2, node3);
#endif
        return quadrangle_area<T, DIM>(node0, node1, node2, node3);
    }
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
cell_volume(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using value_type = T;
    using vector_type = vector_value<value_type>;
    return for_each_simplex_val<value_type, DIM>(nodes, cell_node_index, cell_node_index_start, cell_node_count, 0,
        [](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3){
            return tetrahedron_volume<value_type, DIM>(node0, node1, node2, node3);
        });
}

template<typename T, unsigned DIM>
__HOST __DEVICE
vector_type_t<T, DIM> quadrilateral_mcenter(const vector_type_t<T, DIM> &node0, const vector_type_t<T, DIM> &node1,
    const vector_type_t<T, DIM> &node2, const vector_type_t<T, DIM> &node3)
{
    check_convex_quadrangle<T, DIM>(node0, node1, node2, node3);
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    // ��������� �������������� �� ��� ������������,
    // ������� ������ ���� ���� ������������� � �� ������� (�����),
    // � ����� ������� ����� ���� ���� ������������ ����� � ����� ������� � ���� �������.
    const vector_type
        p1 = (node0 + node1 + node2) / 3.,
        p2 = (node0 + node3 + node2) / 3.;
    const value_type
        S1 = triangle_area<value_type, DIM>(node0, node1, node2),
        S2 = triangle_area<value_type, DIM>(node0, node3, node2),
        S12 = S1 + S2;
    const vector_type mcenter = p1 + (p2 - p1) * (S2 / (S1 + S2));
    const vector_type
        p_1 = (node0 + node1 + node3) / 3.,
        p_2 = (node2 + node1 + node3) / 3.;
    const value_type
        S_1 = triangle_area<value_type, DIM>(node0, node1, node3),
        S_2 = triangle_area<value_type, DIM>(node2, node1, node3),
        S_12 = S_1 + S_2;
    const vector_type mcenter1 = p_1 + (p_2 - p_1) * (S_2 / (S_1 + S_2));
    return mcenter + (mcenter1 - mcenter) * (S_12 / (S12 + S_12));
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector_type_t<T, DIM> >::type
cell_mcenter(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    assert(cell_node_count == 3 || cell_node_count == 4);
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    return cell_node_count == 3 ? (node0 + node1 + node2) / 3. :
        quadrilateral_mcenter<value_type, DIM>(node0, node1, node2, nodes[cell_node_index[cell_node_index_start + 3]]);
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_type_t<T, DIM> >::type
cell_mcenter(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    vector_type mcenter;
    value_type V = 0;
    for_each_simplex<value_type, DIM>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&mcenter, &V](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3)
        {
            const vector_type mcenter1 = (node0 + node1 + node2 + node3) / 4.;
            const value_type V1 = tetrahedron_volume<value_type, DIM>(node0, node1, node2, node3);
            mcenter += (mcenter1 - mcenter) * (V1 / (V += V1));
        });
    return mcenter;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
calc_interface_area(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 2);
    return (nodes[interface_node_index[interface_node_index_start + 1]] - nodes[interface_node_index[interface_node_index_start]]).length();
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
calc_interface_area(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start, unsigned interface_node_count)
{
    assert(interface_node_count == 3 || interface_node_count == 4);
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    const vector_type
        &node0 = nodes[interface_node_index[interface_node_index_start]],
        &node1 = nodes[interface_node_index[interface_node_index_start + 1]],
        &node2 = nodes[interface_node_index[interface_node_index_start + 2]];
    if (interface_node_count == 3)
        return triangle_area<T, DIM>(node0, node1, node2);
    else {
        vector_type const& node3 = nodes[interface_node_index[interface_node_index_start + 3]];
#ifndef NDEBUG
        check_convex_quadrangle<T, DIM>(node0, node1, node2, node3);
#endif
        return quadrangle_area<T, DIM>(node0, node1, node2, node3);
    }
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_min_height(vector2_value<T> const& node0, vector2_value<T> const& node1, vector2_value<T> const& node2)
{
    using value_type = T;
    // ��������� ����������� ������ ������������.
    const value_type
        volume = std::abs((node0 - node1) * (node0 - node2)),
        max_length = func::max(func::max((node0 - node1).length(), (node1 - node2).length()), (node2 - node0).length());
    assert(volume > 0);
    return volume / max_length;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
triangle_height(vector2_value<T> const& node0, vector2_value<T> const& node1, vector2_value<T> const& node2)
{
    using value_type = T;
    // ��������� ������ ������������, ��������� �� ���� node0.
    const value_type
        volume = std::abs((node0 - node1) * (node0 - node2)),
        length = (node1 - node2).length();
    assert(volume > 0);
    return volume / length;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, T>::type
calc_cell_height(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using value_type = T;
    using vector_type = vector2_value<value_type>;
    assert(cell_node_count >= 3);
    const vector_type
        &node0 = nodes[cell_node_index[cell_node_index_start]],
        &node1 = nodes[cell_node_index[cell_node_index_start + 1]],
        &node2 = nodes[cell_node_index[cell_node_index_start + 2]];
    switch (cell_node_count) {
    case 3:
        return triangle_min_height<T, DIM>(node0, node1, node2);
    case 4:
    {
        vector_type const& node3 = nodes[cell_node_index[cell_node_index_start + 3]];
        // ����������� ������ ��������������� ��������� ��� ������� ���� �����.
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
        return value_type();
    }
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_min_height(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    using value_type = T;
    // ��������� ����������� ������ ���������.
    const value_type
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

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
tetrahedron_height(vector_value<T> const& node0, vector_value<T> const& node1,
    vector_value<T> const& node2, vector_value<T> const& node3)
{
    using value_type = T;
    // ��������� ������ ���������, ��������� �� ���� node0.
    const value_type
        volume = std::abs(((node0 - node1) * (node0 - node2)) & (node0 - node3)),
        area = ((node1 - node2) * (node1 - node3)).length();
    assert(volume > 0);
    assert(area > 0);
    return volume / area;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, T>::type
calc_cell_height(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start, unsigned cell_node_count)
{
    using value_type = T;
    using vector_type = vector_type_t<value_type, DIM>;
    value_type height = numeric_limits<value_type>::max();
    for_each_simplex<value_type, DIM>(nodes, cell_node_index, cell_node_index_start, cell_node_count,
        [&height](vector_type const& node0, vector_type const& node1, vector_type const& node2, vector_type const& node3){
            height = func::min(height, tetrahedron_min_height<value_type, DIM>(node0, node1, node2, node3));
        });
    return height;
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 2, vector2_value<T> >::type
calc_normal(vector_proxy<vector2_value<T> > const& nodes,
    vector_proxy<unsigned> const& cell_node_index, unsigned cell_node_index_start)
{
    using vector_type = vector2_value<T>;
    const vector_type vec = nodes[cell_node_index[cell_node_index_start + 1]] - nodes[cell_node_index[cell_node_index_start]];
    return vector_type(vec.value_y(), -vec.value_x());
}

template<typename T, unsigned DIM>
__HOST __DEVICE
typename std::enable_if<DIM == 3, vector_value<T> >::type
calc_normal(vector_proxy<vector_value<T> > const& nodes,
    vector_proxy<unsigned> const& interface_node_index, unsigned interface_node_index_start)
{
    using vector_type = vector_value<T>;
    const vector_type
        &interface_node0 = nodes[interface_node_index[interface_node_index_start]],
        vec1 = nodes[interface_node_index[interface_node_index_start + 1]] - interface_node0,
        vec2 = nodes[interface_node_index[interface_node_index_start + 2]] - interface_node0;
    return vec1 * vec2;
}

_UGRID_MATH_END
