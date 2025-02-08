#pragma once

#include "kiam_math_func.hpp"
#include "grid_expression.hpp"
#include "type_traits.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename GEXP, typename... Args>
struct multi_expr;

template<typename TAG, typename GEXP, typename Arg0, typename... Args>
struct multi_expr<TAG, GEXP, Arg0, Args...> : multi_expr<TAG, GEXP, Args...>
{
    static_assert(std::is_same<TAG, typename Arg0::tag_type>::value, "Tag types should be the same");

    using super = multi_expr<TAG, GEXP, Args...>;

    multi_expr(GRID_EXPR(Arg0) const& arg0, GRID_EXPR(Args) const&... args) : super(args...), arg0_proxy(arg0.get_proxy()){}

    template<typename... Args2>
    __DEVICE auto eval(size_t i, Args2 const&... args2) const {
        return super::eval(i, args2..., arg0_proxy);
    }

private:
    typename Arg0::proxy_type const arg0_proxy;
};

template<typename TAG, typename GEXP>
struct multi_expr<TAG, GEXP> : grid_expression<TAG, GEXP>
{
    template<typename... Args>
    __DEVICE auto eval(size_t i, Args const&... args) const -> decltype(GEXP::apply(args[i]...)){
        return GEXP::apply(args[i]...);
    }
};

template<typename GEXP, typename Arg0, typename... Args>
struct multi_var_fun : multi_expr<typename Arg0::tag_type, GEXP, Arg0, Args...>
{
    using super = multi_expr<typename Arg0::tag_type, GEXP, Arg0, Args...>;
    using value_type = typename Arg0::value_type;

    multi_var_fun(GRID_EXPR(Arg0) const& args0, GRID_EXPR(Args) const&... args) : super(args0, args...){}

    __DEVICE auto operator[](size_t i) const -> decltype(super::eval(i)){
        return super::eval(i);
    }
};

#define DECLARE_FUNC_EVAL_OBJ(func_name, func_impl) \
    template<typename... Args> \
    struct func_name##_func_grid_expression : multi_var_fun<func_name##_func_grid_expression<Args...>, Args...>{ \
        func_name##_func_grid_expression(GRID_EXPR(Args) const&... args) : multi_var_fun<func_name##_func_grid_expression<Args...>, Args...>(args...){} \
        template<typename... Args2> \
        __DEVICE static auto apply(Args2 const&... args) -> decltype(func_impl(args...)){ \
            return func_impl(args...); \
        } \
    }; \
    template<typename... Args> \
    func_name##_func_grid_expression<Args...> func_name(GRID_EXPR(Args) const&... args){ \
        return func_name##_func_grid_expression<Args...>(args...); \
    }

#define DECLARE_STD_FUNC_EVAL_OBJ(name) DECLARE_FUNC_EVAL_OBJ(name, name)

DECLARE_STD_FUNC_EVAL_OBJ(sqrt)
DECLARE_STD_FUNC_EVAL_OBJ(sin)
DECLARE_STD_FUNC_EVAL_OBJ(cos)
DECLARE_STD_FUNC_EVAL_OBJ(tan)
DECLARE_STD_FUNC_EVAL_OBJ(asin)
DECLARE_STD_FUNC_EVAL_OBJ(acos)
DECLARE_STD_FUNC_EVAL_OBJ(atan)
DECLARE_STD_FUNC_EVAL_OBJ(sinh)
DECLARE_STD_FUNC_EVAL_OBJ(cosh)
DECLARE_STD_FUNC_EVAL_OBJ(tanh)
DECLARE_STD_FUNC_EVAL_OBJ(ceil)
DECLARE_STD_FUNC_EVAL_OBJ(floor)
DECLARE_STD_FUNC_EVAL_OBJ(exp)
DECLARE_STD_FUNC_EVAL_OBJ(log)
DECLARE_STD_FUNC_EVAL_OBJ(log10)

#ifdef __CUDACC__
DECLARE_STD_FUNC_EVAL_OBJ(sinpi)
DECLARE_STD_FUNC_EVAL_OBJ(cospi)
DECLARE_STD_FUNC_EVAL_OBJ(exp2)
DECLARE_STD_FUNC_EVAL_OBJ(log1p)
DECLARE_STD_FUNC_EVAL_OBJ(log2)
#endif  // __CUDACC__

DECLARE_FUNC_EVAL_OBJ(sqr, func::sqr)

_KIAM_MATH_END
