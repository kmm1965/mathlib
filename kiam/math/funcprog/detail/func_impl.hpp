#pragma once

_FUNCPROG_BEGIN

// Function application
template<typename Ret, typename Arg0, typename... Args>
function_t<Ret(Args...)> operator<<(
	function_t<Ret(Arg0, Args...)> const& f,
	fdecay<Arg0> const& arg0)
{
	return [f, arg0](Args... args) { return f(arg0, args...); };
}

// Application to the function without parameters
template<typename Ret, typename Arg0, typename... Args>
function_t<Ret(Args...)> operator<<(
	function_t<Ret(Arg0, Args...)> const& f,
	f0<fdecay<Arg0> > const& arg0)
{
	return [f, arg0](Args... args) { return f(arg0(), args...); };
}

// Function dereference - same as func()
template<typename Ret>
Ret operator*(f0<Ret> const& f) {
	return f();
}

template<typename T>
T deref(f0<T> const& f) {
	return f();
}

_FUNCPROG_END

namespace std {

template<typename Ret>
ostream& operator<<(ostream &os, _FUNCPROG::f0<Ret> const& f) {
	return os << f();
}

template<typename Ret>
wostream& operator<<(wostream &os, _FUNCPROG::f0<Ret> const& f) {
	return os << f();
}

template<typename T1, typename T2>
ostream& operator<<(ostream &os, const pair<T1, T2> &p) {
    return os << '(' << p.first << ',' << p.second << ')';
}

template<typename T1, typename T2>
wostream& operator<<(wostream &os, const pair<T1, T2> &p) {
    return os << L'(' << p.first << L',' << p.second << L')';
}

template<typename T1, typename T2>
ostream& operator<<(ostream &os, _FUNCPROG::f0<pair<T1, T2> > const& f) {
    return os << f();
}

template<typename T1, typename T2>
wostream& operator<<(wostream &os, _FUNCPROG::f0<pair<T1, T2> > const& f) {
    return os << f();
}

template<typename T1, typename T2, typename T3>
ostream& operator<<(ostream &os, const tuple<T1, T2, T3> &t) {
    return os << '(' << get<0>(t) << ',' << get<1>(t) << ',' << get<2>(t) << ')';
}

template<typename T1, typename T2, typename T3>
wostream& operator<<(wostream &os, const tuple<T1, T2, T3> &t) {
    return os << L'(' << get<0>(t) << L',' << get<1>(t) << L',' << get<2>(t) << L')';
}

template<typename T1, typename T2, typename T3>
ostream& operator<<(ostream &os, _FUNCPROG::f0<tuple<T1, T2, T3> > const& f) {
    return os << f();
}

template<typename T1, typename T2, typename T3>
wostream& operator<<(wostream &os, _FUNCPROG::f0<tuple<T1, T2, T3> > const& f) {
    return os << f();
}

} // namespace std
