#pragma once

_FUNCPROG_BEGIN

template<typename Ret, typename T, typename... Args>
void map_impl(List<remove_f0_t<function_t<Ret(Args...)> > > &result,
	function_t<Ret(T, Args...)> const& f, List<fdecay<T> > const& l)
{
    std::transform(l.cbegin(), l.cend(), std::back_inserter(result),
        [&f](fdecay<T> const& v) { return invoke_f0(f << v); });
}

DEFINE_FUNCTION_2_ARGS(2, List<remove_f0_t<function_t<T0(Args...)> > >, map, function_t<T0(T1, Args...)> const&, f, List<fdecay<T1> > const&, l,
	List<remove_f0_t<function_t<T0(Args...)> > > result;
    map_impl(result, f, l);
	return result;)

_FUNCPROG_END
