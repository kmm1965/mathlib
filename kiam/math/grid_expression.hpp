#pragma once

#include "context.hpp"

_KIAM_MATH_BEGIN

template<typename TAG>
struct _grid_expression
{
    using tag_type = TAG;
    using base_class = _grid_expression;

protected:
    constexpr _grid_expression(){} // Protect from direct construction
};

template<typename TAG, class GEXP, class _Proxy = GEXP>
struct grid_expression : math_object<GEXP, _Proxy>
{
    using tag_type = TAG;

    template<typename T>
    __DEVICE
    void assign(T &val, size_t i) const {
        val = (*this)()[i];
    }

    template<typename T, typename CONTEXT>
    __DEVICE
    void assign(T &val, size_t i, context<CONTEXT> const& context) const {
        val = (*this)()(i, context);
    }

protected:
    constexpr grid_expression(){} // Protect from direct construction
};

template<typename TAG, typename T, typename F>
struct inplace_grid_expression : grid_expression<TAG, inplace_grid_expression<TAG, T, F> >
{
    typedef T value_type;

    inplace_grid_expression(F f) : f(f){}

    __DEVICE
    value_type operator()(size_t i) const {
        return f(i);
    }

    template<typename CONTEXT>
    __DEVICE
    value_type operator()(size_t i, context<CONTEXT> const& ctx) const {
        return f(i, ctx());
    }

private:
    F const f;
};

template<typename TAG, typename T, typename F>
inplace_grid_expression<TAG, T, F> get_inplace_grid_expression(F f){
    return inplace_grid_expression<TAG, T, F>(f);
}

_KIAM_MATH_END

#define GRID_EXPR(GEXP) _KIAM_MATH::grid_expression<typename GEXP::tag_type, GEXP, typename GEXP::proxy_type>

#include "func_grid_expression.hpp"

#ifdef DONT_USE_CXX_11
#define DECLARE_MATH_GRID_EXPRESSION(name) \
    template<class GEXP, class _Proxy = GEXP> \
    struct name##_grid_expression : _KIAM_MATH::grid_expression<name##_tag, GEXP, _Proxy>{}
#else
#define DECLARE_MATH_GRID_EXPRESSION(name) \
    template<class GEXP, class _Proxy = GEXP> \
    using name##_grid_expression = _KIAM_MATH::grid_expression<name##_tag, GEXP, _Proxy>
#endif
