#pragma once

_FUNCPROG2_BEGIN

// comma
DECLARE_FUNCTION_2(2, PAIR_T(T0, T1), comma, T0 const&, T1 const&)
FUNCTION_TEMPLATE(2) constexpr PAIR_T(T0, T1) comma(T0 const& a, T1 const& b) {
    return PAIR_T(T0, T1)(a, b);
}

// fst
template<typename T0, typename T1>
T0 fst(pair_t<T0, T1> const & p) {
    return p.first;
}

// snd
template<typename T0, typename T1>
T1 snd(pair_t<T0, T1> const & p) {
    return p.second;
}

// curry
DECLARE_FUNCTION_3(4, T2, curry, FUNCTION2(T2(const pair_t<T0, T1>&), T3) const&, T0 const&, T1 const&)
FUNCTION_TEMPLATE(4) constexpr T2 curry(FUNCTION2(T2(const pair_t<T0, T1>&), T3) const& f, T0 const& v1, T1 const& v2) {
    return f(std::make_pair(v1, v2));
}

// uncurry
DECLARE_FUNCTION_2(4, T2, uncurry, FUNCTION2(T2(T0, T1), T3) const&, PAIR_T(fdecay<T0>, fdecay<T1>) const&)
FUNCTION_TEMPLATE(4) constexpr T2 uncurry(FUNCTION2(T2(T0, T1), T3) const& f, PAIR_T(fdecay<T0>, fdecay<T1>) const& p) {
    return f(p.first, p.second);
}

_FUNCPROG2_END
