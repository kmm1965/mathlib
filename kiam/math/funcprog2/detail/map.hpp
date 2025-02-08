#pragma once

_FUNCPROG2_BEGIN

template<typename Ret, typename FuncImpl, typename T>
__DEVICE void map_impl(List<Ret> &result, function2<Ret(T), FuncImpl> const& f, List<fdecay<T> > const& l){
    std::transform(std::cbegin(l), std::cend(l), std::back_inserter(result), [&f](fdecay<T> const& v) { return f(v)); });
}

DECLARE_FUNCTION_2(3, List<remove_f0_t<function2<T0(Args...)> > >, map, FUNCTION2(T0(T1), T2) const&, List<fdecay<T1> > const&)
FUNCTION_TEMPLATE(3) __DEVICE constexpr List<T0> map(FUNCTION2<T0(T1), T2) const& f, List<fdecay<T1> > const& l)
{
    List<T0> result;
    map_impl(result, f, l);
    return result;
}

_FUNCPROG2_END
