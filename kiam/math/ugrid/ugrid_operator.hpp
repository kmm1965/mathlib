#pragma once

#include "ugrid_def.h"

_UGRID_MATH_BEGIN

template<class G, class GO, class _Proxy = GO>
struct ugrid_operator1 : ugrid_operator<GO, _Proxy>
{
    typedef G grid_type;

protected:
    ugrid_operator1(grid_type const& grid) : m_grid(grid.get_proxy()){}

public:
    __DEVICE __HOST
    typename grid_type::proxy_type const& get_grid() const { return m_grid; }

private:
    typename grid_type::proxy_type const m_grid;
};

template<class G, typename F>
struct inplace_ugrid_operator : ugrid_operator1<G, inplace_ugrid_operator<G, F> >
{
    using super = ugrid_operator1<G, inplace_ugrid_operator>;

    inplace_ugrid_operator(grid_type const& grid, F f) : super(grid), f(f){}

    template<typename EOP>
    __DEVICE __HOST
    auto operator()(int i, int j, EOP const& eobj_proxy) const {
        return f(i, j, eobj_proxy);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(inplace_ugrid_operator)

private:
    F f;
};

template<class G, typename F>
inplace_ugrid_operator<G, F> get_inplace_ugrid_operator(G const& grid, F f){
    return inplace_ugrid_operator<G, F>(grid, f);
}

_UGRID_MATH_END
