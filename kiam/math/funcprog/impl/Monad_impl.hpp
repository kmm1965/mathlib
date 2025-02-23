#pragma once

#include "../Monad.hpp"
#include "../Applicative.hpp"
#include "../MonadFail.hpp"

_FUNCPROG_BEGIN

template<typename _M>
template<typename A>
constexpr typeof_t<_M, A> _Monad<_M>::mreturn(A const& x){
    return Applicative<_M>::pure(x);
}

template<typename _M>
template<typename A>
constexpr typeof_t<_M, A> _Monad<_M>::fail(const char* msg){
    return MonadFail<_M>::template fail<A>(msg);
}

template<typename _M>
template<typename A>
constexpr typeof_t<_M, A> _Monad<_M>::fail(std::string const& msg){
    return fail<A>(msg.c_str());
}

template<typename MF, typename MG>
constexpr same_monad_type<MF, MG, MG> operator>>(MF const& f, MG const& g){
    return _do(__unused__, f, return g;);
}

/*
-- | Promote a function to a Monad.
liftM   :: (Monad m) => (a1 -> r) -> m a1 -> m r
liftM f m1              = do { x1 <- m1; return (f x1) }

-- | Promote a function to a Monad, scanning the monadic arguments from
-- left to right.  For example,
--
-- > liftM2 (+) [0,1] [0,2] = [0,2,1,3]
-- > liftM2 (+) (Just 1) Nothing = Nothing
--
liftM2  :: (Monad m) => (a1 -> a2 -> r) -> m a1 -> m a2 -> m r
liftM2 f m1 m2          = do { x1 <- m1; x2 <- m2; return (f x1 x2) }
-- Caution: since this may be used for `liftA2`, we can't use the obvious
-- definition of liftM2 = liftA2.

-- | Promote a function to a Monad, scanning the monadic arguments from
-- left to right (cf. 'liftM2').
liftM3  :: (Monad m) => (a1 -> a2 -> a3 -> r) -> m a1 -> m a2 -> m a3 -> m r
liftM3 f m1 m2 m3       = do { x1 <- m1; x2 <- m2; x3 <- m3; return (f x1 x2 x3) }

-- | Promote a function to a Monad, scanning the monadic arguments from
-- left to right (cf. 'liftM2').
liftM4  :: (Monad m) => (a1 -> a2 -> a3 -> a4 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m r
liftM4 f m1 m2 m3 m4    = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; return (f x1 x2 x3 x4) }

-- | Promote a function to a Monad, scanning the monadic arguments from
-- left to right (cf. 'liftM2').
liftM5  :: (Monad m) => (a1 -> a2 -> a3 -> a4 -> a5 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m a5 -> m r
liftM5 f m1 m2 m3 m4 m5 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; x5 <- m5; return (f x1 x2 x3 x4 x5) }
*/

// liftM f m1 = do { x1 <- m1; return (f x1) }

FUNCTION_TEMPLATE_ARGS(3) constexpr LIFTM_TYPE(T0, T1, T2) liftM(function_t<T1(T2, Args...)> const& f, T0 const& m) {
    return _do(x, m, return Monad_t<T0>::mreturn(invoke_f0(f << x)););
}

// liftM2 :: Monad m => (a1 -> a2 -> r) -> m a1 -> m a2 -> m r
FUNCTION_TEMPLATE(5) constexpr LIFTM2_TYPE(T0, T1, T2, T3, T4) liftM2(function_t<T2(T3, T4)> const& f, T0 const& m0, T1 const& m1) {
    return _do2(x0, m0, x1, m1, return Monad_t<T0>::mreturn(f(x0, x1)););
}

// liftM3 :: Monad m => (a1 -> a2 -> a3 -> r) -> m a1 -> m a2 -> m a3 -> m r 
FUNCTION_TEMPLATE(7) constexpr LIFTM3_TYPE(T0, T1, T2, T3, T4, T5, T6) liftM3(function_t<T3(T4, T5, T6)> const& f,
    T0 const& m0, T1 const& m1, T2 const& m2)
{
    return _do3(x0, m0, x1, m1, x2, m2, return Monad_t<T0>::mreturn(f(x0, x1, x2)););
}

// liftM4 :: Monad m => (a1 -> a2 -> a3 -> a4 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m r 
FUNCTION_TEMPLATE(9) constexpr LIFTM4_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8) liftM4(
    function_t<T4(T5, T6, T7, T8)> const& f, T0 const& m0, T1 const& m1, T2 const& m2, const T3& m3)
{
    return _do4(x0, m0, x1, m1, x2, m2, x3, m3, return Monad_t<T0>::mreturn(f(x0, x1, x2, x3)););
}

// liftM5 :: Monad m => (a1 -> a2 -> a3 -> a4 -> a5 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m a5 -> m r 
// liftM5 f m1 m2 m3 m4 m5 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; x5 <- m5; return (f x1 x2 x3 x4 x5) }
FUNCTION_TEMPLATE(11) constexpr LIFTM5_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) liftM5(
    function_t<T5(T6, T7, T8, T9, T10)> const& f,
    T0 const& m0, T1 const& m1, T2 const& m2, const T3& m3, const T4& m4)
{
    return _do5(x0, m0, x1, m1, x2, m2, x3, m3, x4, m4, return Monad_t<T0>::mreturn(f(x0, x1, x2, x3, x4)););
}

/*
    In many situations, the 'liftM' operations can be replaced by uses of
'ap', which promotes function application.

return f `ap` x1 `ap` ... `ap` xn
is equivalent to
liftMn f x1 x2 ... xn

ap                :: (Monad m) => m (a -> b) -> m a -> m b
ap m1 m2          = do { x1 <- m1; x2 <- m2; return (x1 x2) }
*/
// ap       :: (Monad m) => m (a -> b) -> m a -> m b
// ap m1 m2 = do { x1 <- m1; x2 <- m2; return (x1 x2) }
FUNCTION_TEMPLATE(2) constexpr AP_TYPE(T0, T1) ap(T1 const& mf, T0 const& mv) {
    return _do2(f, mf, v, mv, return Monad_t<T0>::mreturn(invoke_f0(f << v)););
}

_FUNCPROG_END
