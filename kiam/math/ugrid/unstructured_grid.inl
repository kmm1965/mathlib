#pragma once

#include "unstructured_grid.hpp"
#include "../vector_proxy.hpp"

_UGRID_MATH_BEGIN

template<typename T, unsigned DIM>
struct unstructured_grid_proxy
{
    typedef T value_type;
    typedef unstructured_grid<value_type, DIM> grid_type;
    typedef typename grid_type::vector_t vector_t;
    typedef typename grid_type::cell_value_type cell_value_type;
    typedef typename grid_type::cell_b_value_type cell_b_value_type;
    typedef typename grid_type::interface_value_type interface_value_type;

    unstructured_grid_proxy(grid_type const& grid) :
        m_nodes(grid.m_nodes), m_nodes_l2g(grid.m_nodes_l2g), m_cells(grid.m_cells), m_cells_l2g(grid.m_cells_l2g),
        m_cells_b(grid.m_cells_b), m_interfaces(grid.m_interfaces),
        m_cell_node_shift(grid.m_cell_node_shift), m_cell_node_index(grid.m_cell_node_index),
        m_cell_interface_shift(grid.m_cell_interface_shift), m_cell_interface_index(grid.m_cell_interface_index),
        m_interface_node_shift(grid.m_interface_node_shift), m_interface_node_index(grid.m_interface_node_index),
        m_ptMin(grid.m_ptMin), m_ptMax(grid.m_ptMax), m_dxMin(grid.m_dxMin),
#if UGRID_CELL_H
        m_hMin(grid.m_hMin),
#endif
        m_local_cells(grid.m_local_cells), m_global_cells(grid.m_global_cells),
        m_local_nodes(grid.m_local_nodes), m_global_nodes(grid.m_global_nodes){}

    __DEVICE
    bool is_external_normal(unsigned iface, size_t icell) const {
        assert(m_interfaces[iface].cells[0] == icell || m_interfaces[iface].cells[1] == icell);
        return m_interfaces[iface].cells[0] == icell;
    }

    _KIAM_MATH::vector_proxy<vector_t> const m_nodes;   // массив координат узлов
    _KIAM_MATH::vector_proxy<unsigned> const m_nodes_l2g;
    _KIAM_MATH::vector_proxy<cell_value_type> const m_cells;
    _KIAM_MATH::vector_proxy<unsigned> const m_cells_l2g;
    _KIAM_MATH::vector_proxy<cell_b_value_type> const m_cells_b;
    _KIAM_MATH::vector_proxy<interface_value_type> const m_interfaces;
    const _KIAM_MATH::vector_proxy<unsigned> m_cell_node_shift, m_cell_node_index, m_cell_interface_shift, m_cell_interface_index,
        m_interface_node_shift, m_interface_node_index;
    const vector_t m_ptMin, m_ptMax;    // Минимальные и максимальные координаты вершин.
    const vector_t m_dxMin; // Минимальные размеры ячеек.
#if UGRID_CELL_H
    value_type const m_hMin;        // Минимальный характерный размер ячейки.
#endif
    const unsigned m_local_cells, m_global_cells, m_local_nodes, m_global_nodes;
};

_UGRID_MATH_END
