#pragma once

#include "../dense_grid_function.hpp"
#include "ugrid_def.h"

_UGRID_MATH_BEGIN

enum ugrid_function_type { empty_grid_function = -1, on_inner_cells, on_local_inner_cells, on_boundary_cells, on_all_cells, on_faces };

template<typename T>
struct dense_ugrid_function : _KIAM_MATH::dense_grid_function<ugrid_tag, T>
{
    typedef T value_type;
    typedef _KIAM_MATH::dense_grid_function<ugrid_tag, value_type> super;

    void operator=(const host_vector<value_type>& v) { super::assign(v); }

    template<class G>
    dense_ugrid_function(const G &grid, ugrid_function_type type = on_inner_cells, const value_type &init = value_type()) :
        super(type == empty_grid_function ? 0 : type == on_inner_cells ? grid.m_cells.size() :
        type == on_local_inner_cells ? grid.m_local_cells :
        type == on_boundary_cells ? grid.m_cells_b.size() :
        type == on_all_cells ? grid.m_cells.size() + grid.m_cells_b.size() : grid.m_interfaces.size(),
        grid.m_local_cells, init){}

    REIMPLEMENT_GRID_FUNCTION_OPERATORS()
};

_UGRID_MATH_END
