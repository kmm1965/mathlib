#pragma once

_FUNCPROG_BEGIN

//void _Monad::fail(const char *msg){
//  throw MonadError(msg);
//}

template<typename M>
join_type<M> join(M const& x) {
    return x >>= id<typename M::value_type>();
}

// liftM :: Monad m => (a1 -> r) -> m a1 -> m r
//template<typename M, typename Ret, typename Arg, typename... Args>
//liftM_type<M, Ret, Arg, Args...>
//liftM(function_t<Ret(Arg, Args...)> const& f, M const& m) {
//  return _do(x, m, return Monad_t<M>::mreturn(invoke_f0(f << x)););
//}

DEFINE_FUNCTION_2_ARGS(3, LIFTM_TYPE(T0, T1, T2), liftM, function_t<T1(T2, Args...)> const&, f, T0 const&, m,
    return _do(x, m, return Monad_t<T0>::mreturn(invoke_f0(f << x)););)

// liftM2 :: Monad m => (a1 -> a2 -> r) -> m a1 -> m a2 -> m r
DEFINE_FUNCTION_3(5, LIFTM2_TYPE(T0, T1, T2, T3, T4), liftM2, function_t<T2(T3, T4)> const&, f, T0 const&, m0, T1 const&, m1,
    return _do2(x0, m0, x1, m1, return Monad_t<T0>::mreturn(f(x0, x1)););)

// liftM3 :: Monad m => (a1 -> a2 -> a3 -> r) -> m a1 -> m a2 -> m a3 -> m r 
DEFINE_FUNCTION_4(7, LIFTM3_TYPE(T0, T1, T2, T3, T4, T5, T6), liftM3, function_t<T3(T4, T5, T6)> const&, f,
    T0 const&, m0, T1 const&, m1, T2 const&, m2,
        return _do3(x0, m0, x1, m1, x2, m2, return Monad_t<T0>::mreturn(f(x0, x1, x2)););)

// liftM4 :: Monad m => (a1 -> a2 -> a3 -> a4 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m r 
DEFINE_FUNCTION_5(9, LIFTM4_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8), liftM4,
    function_t<T4(T5, T6, T7, T8)> const&, f, T0 const&, m0, T1 const&, m1, T2 const&, m2, const T3&, m3,
        return _do4(x0, m0, x1, m1, x2, m2, x3, m3, return Monad_t<T0>::mreturn(f(x0, x1, x2, x3)););)

// liftM5 :: Monad m => (a1 -> a2 -> a3 -> a4 -> a5 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m a5 -> m r 
DEFINE_FUNCTION_6(11, LIFTM5_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10), liftM5,
    function_t<T5(T6, T7, T8, T9, T10)> const&, f,
    T0 const&, m0, T1 const&, m1, T2 const&, m2, const T3&, m3, const T4&, m4,
        return _do5(x0, m0, x1, m1, x2, m2, x3, m3, x4, m4, return Monad_t<T0>::mreturn(f(x0, x1, x2, x3, x4)););)

// ap       :: (Monad m) => m (a -> b) -> m a -> m b
// ap m1 m2 = do { x1 <- m1; x2 <- m2; return (x1 x2) }
DEFINE_FUNCTION_2(2, AP_TYPE(T0, T1), ap, T1 const&, mf, T0 const&, mv,
    return _do2(f, mf, v, mv, return Monad_t<T0>::mreturn(invoke_f0(f << v)););)

template<typename MF, typename MG>
constexpr same_monad_type<MF, MG, MG> operator>>(MF const& f, MG const& g) {
    return _do(__unused__, f, return g;);
}

template<typename RetF, typename RetG>
constexpr same_monad_type<RetF, RetG, f0<RetG> > operator>>(f0<RetF> const& f, f0<RetG> const& g) {
    return [f, g](){ return _do(__unused__, f(), return g();); };
}

// >=>
template<typename RetG, typename ArgG, typename RetF, typename... ArgsF>
constexpr typename std::enable_if<
    is_same_monad<RetF, RetG>::value && is_same_as<ArgG, typename RetF::value_type>::value,
    function_t<RetG(ArgsF...)>
>::type operator^(function_t<RetF(ArgsF...)> const& f, function_t<RetG(ArgG)> const& g) {
    return [g, f](ArgsF... args) { return f(args...) >>= g; };
}

_FUNCPROG_END
