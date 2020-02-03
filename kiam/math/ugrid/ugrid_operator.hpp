#pragma once

#include "ugrid_def.h"

_UGRID_MATH_BEGIN

template<class G, class GO, class _Proxy = GO>
struct ugrid_operator1 : ugrid_operator<GO, _Proxy>
{
    typedef G grid_type;

protected:
    ugrid_operator1(const grid_type &grid) : m_grid(grid.get_proxy()){}

public:
    __DEVICE __HOST
    const typename grid_type::proxy_type &get_grid() const { return m_grid; }

private:
    const typename grid_type::proxy_type m_grid;
};

_UGRID_MATH_END
