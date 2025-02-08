#pragma once

#include "fwd/Monad_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _M>
struct _Monad // Default implementation of some functions
{
    template<typename A>
    static constexpr typeof_t<_M, A> mreturn(A const& x);

    template<typename A = None>
    static constexpr typeof_t<_M, A> fail(const char* msg);

    template<typename A = None>
    static constexpr typeof_t<_M, A> fail(std::string const& msg);

    //-- | Promote a function to a Monad.
    //liftM   :: (Monad m) => (a1 -> r) -> m a1 -> m r
    //liftM f m1              = do { x1 <- m1; return (f x1) }
    template<typename Ret, typename Arg, typename... Args>
    static constexpr auto liftM(function_t<Ret(Arg, Args...)> const& f){
        return _([f](typeof_dt<_M, Arg> const& m){
            return _do(x, m,
                return Monad<_M>::mreturn(invoke_f0(f << x)););
        });
    }

    //-- | Promote a function to a Monad, scanning the monadic arguments from
    //-- left to right.  For example,
    //--
    //-- > liftM2 (+) [0,1] [0,2] = [0,2,1,3]
    //-- > liftM2 (+) (Just 1) Nothing = Nothing
    //--
    //liftM2  :: (Monad m) => (a1 -> a2 -> r) -> m a1 -> m a2 -> m r
    //liftM2 f m1 m2          = do { x1 <- m1; x2 <- m2; return (f x1 x2) }
    //-- Caution: since this may be used for `liftA2`, we can't use the obvious
    //-- definition of liftM2 = liftA2.
    template<typename Ret, typename Arg1, typename Arg2, typename... Args>
    static constexpr auto liftM2(function_t<Ret(Arg1, Arg2, Args...)> const& f){
        return _([f](typeof_dt<_M, Arg1> const& m1, typeof_dt<_M, Arg2> const& m2){
            return _do2(x1, m1, x2, m2,
                return Monad<_M>::mreturn(invoke_f0(f << x1 << x2)););
        });
    }

    //-- | Promote a function to a Monad, scanning the monadic arguments from
    //-- left to right (cf. 'liftM2').
    //liftM3  :: (Monad m) => (a1 -> a2 -> a3 -> r) -> m a1 -> m a2 -> m a3 -> m r
    //liftM3 f m1 m2 m3       = do { x1 <- m1; x2 <- m2; x3 <- m3; return (f x1 x2 x3) }
    template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename... Args>
    static constexpr auto liftM3(function_t<Ret(Arg1, Arg2, Arg3, Args...)> const& f){
        return _([f](typeof_dt<_M, Arg1> const& m1, typeof_dt<_M, Arg2> const& m2, typeof_dt<_M, Arg3> const& m3){
            return _do3(x1, m1, x2, m2, x3, m3,
                return Monad<_M>::mreturn(invoke_f0(f << x1 << x2 << x3)););
        });
    }

    //-- | Promote a function to a Monad, scanning the monadic arguments from
    //-- left to right (cf. 'liftM2').
    //liftM4  :: (Monad m) => (a1 -> a2 -> a3 -> a4 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m r
    //liftM4 f m1 m2 m3 m4    = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; return (f x1 x2 x3 x4) }
    template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename... Args>
    static constexpr auto liftM4(function_t<Ret(Arg1, Arg2, Arg3, Arg4, Args...)> const& f){
        return _([f](typeof_dt<_M, Arg1> const& m1, typeof_dt<_M, Arg2> const& m2,
            typeof_dt<_M, Arg3> const& m3, typeof_dt<_M, Arg4> const& m4)
        {
            return _do4(x1, m1, x2, m2, x3, m3, x4, m4,
                return Monad<_M>::mreturn(invoke_f0(f << x1 << x2 << x3 << x4)););
        });
    }

    //-- | Promote a function to a Monad, scanning the monadic arguments from
    //-- left to right (cf. 'liftM2').
    //liftM5  :: (Monad m) => (a1 -> a2 -> a3 -> a4 -> a5 -> r) -> m a1 -> m a2 -> m a3 -> m a4 -> m a5 -> m r
    //liftM5 f m1 m2 m3 m4 m5 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; x5 <- m5; return (f x1 x2 x3 x4 x5) }
    template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename... Args>
    static constexpr auto liftM5(function_t<Ret(Arg1, Arg2, Arg3, Arg4, Arg5, Args...)> const& f){
        return _([f](typeof_dt<_M, Arg1> const& m1, typeof_dt<_M, Arg2> const& m2,
            typeof_dt<_M, Arg3> const& m3, typeof_dt<_M, Arg4> const& m4, typeof_dt<_M, Arg5> const& m5)
        {
            return _do5(x1, m1, x2, m2, x3, m3, x4, m4, x5, m5,
                return Monad<_M>::mreturn(invoke_f0(f << x1 << x2 << x3 << x4 << x5)););
        });
    }
};

#define DECLARE_MONAD_CLASS(M, _M) \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr remove_f0_t<function_t<M<Ret>(Args...)> > \
    mbind(M<fdecay<Arg> > const& m, function_t<M<Ret>(Arg, Args...)> const& f); \
    template<typename T> using liftM_type = _M::template type<T>;

_FUNCPROG_END
