#pragma once

#include "grid_expression.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class GEXP>
struct negate_grid_expression : grid_expression<typename GEXP::tag_type, negate_grid_expression<GEXP> >
{
    typedef typename GEXP::tag_type tag_type;
    typedef GRID_EXPR(GEXP) gexp_type;
    typedef get_value_type_t<GEXP> value_type;

    negate_grid_expression(const gexp_type& gexp) : gexp_proxy(gexp.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return -gexp_proxy[i];
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return -gexp_proxy(i);
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, context<CONTEXT> const& context) const {
        return -gexp_proxy(i, context);
    }

private:
    typename GEXP::proxy_type const gexp_proxy;
};

template<class GEXP>
negate_grid_expression<GEXP> operator-(GRID_EXPR(GEXP) const& gexp){
    return negate_grid_expression<GEXP>(gexp);
}

_KIAM_MATH_END
