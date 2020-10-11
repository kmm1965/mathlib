#pragma once

#include "Applicative.hpp"

_FUNCPROG_BEGIN

class monad_error : std::runtime_error
{
public:
    explicit monad_error(std::string const& _Message) : runtime_error(_Message){}
    explicit monad_error(const char* _Message) : runtime_error(_Message){}
};

template<class _M>
struct _is_monad : std::false_type {};

template<class M>
using is_monad = _is_monad<base_class_t<M> >;

template<class M, typename T = M>
using monad_type = typename std::enable_if<is_monad<M>::value, T>::type;

template<class _M1, class _M2>
struct _is_same_monad : std::false_type {};

template<class _M>
struct _is_same_monad<_M, _M> : _is_monad<_M> {};

template<class M1, class M2>
using is_same_monad = _is_same_monad<base_class_t<M1>, base_class_t<M2> >;

template<class M1, class M2, typename T>
using same_monad_type = typename std::enable_if<is_same_monad<M1, M2>::value, T>::type;

// Requires mreturn and >>=
template<typename M>
struct Monad;

template<typename T>
using Monad_t = Monad<base_class_t<T> >;

#define DECLARE_MONAD_CLASS(M, _M) \
    template<typename T> \
    static constexpr M<fdecay<T> > mreturn(T const& x); \
    template<typename Ret, typename Arg, typename... Args> \
    static remove_f0_t<function_t<M<Ret>(Args...)> > \
    constexpr mbind(M<fdecay<Arg> > const& m, function_t<M<Ret>(Arg, Args...)> const& f); \
    template<typename T> \
    using liftM_type = _M::template type<T>;

#define IMPLEMENT_MONAD(_M) \
    template<> struct _is_monad<_M> : std::true_type {}

#define IMPLEMENT_MRETURN(M, _M) \
    template<typename T> \
    constexpr M<fdecay<T> > Monad<_M>::mreturn(T const& x){ return super::pure(x);  }

template<typename M>
using join_type = typename std::enable_if<
    is_same_monad<M, typename M::value_type>::value,
    typename M::value_type
>::type;

template<typename M>
join_type<M> join(M const& v);

template<typename MF, typename MG>
constexpr same_monad_type<MF, MG, MG> operator>>(MF const& f, MG const& g);

template<typename RetF, typename RetG>
constexpr same_monad_type<RetF, RetG, f0<RetG> > operator>>(f0<RetF> const& f, f0<RetG> const& g);

// >=>
template<typename RetG, typename ArgG, typename RetF, typename... ArgsF>
constexpr typename std::enable_if<
    is_same_monad<RetF, RetG>::value && is_same_as<ArgG, typename RetF::value_type>::value,
    function_t<RetG(ArgsF...)>
>::type operator^(function_t<RetF(ArgsF...)> const& f, function_t<RetG(ArgG)> const& g);

template<class M, class MFUNC>
struct mbind_result_type;

template<class M, typename MFUNC>
using mbind_result_type_t = typename mbind_result_type<M, MFUNC>::type;

template<class M, class MF, typename Arg, typename... Args>
struct mbind_result_type<M, function_t<MF(Arg, Args...)> >
{
    static_assert(is_same_monad<M, MF>::value, "Should be the same Monad");
    static_assert(is_same_as<value_type_t<M>, Arg>::value, "Should be the same");

    using type = remove_f0_t<function_t<typeof_t<M, value_type_t<MF> >(Args...)> >;
};

template<class M, class MF, typename Arg, typename... Args>
struct mbind_result_type<f0<M>, function_t<MF(Arg, Args...)> >
{
    static_assert(is_same_monad< base_class_t<M>, base_class_t<MF> >::value, "Should be the same Monad");
    static_assert(is_same_as<value_type_t<M>, Arg>::value, "Should be the same");

    using type = typename M::template type<remove_f0_t<function_t<value_type_t<MF>(Args...)> > >;
};

template<typename M, typename MFUNC>
using mbind_type = typename std::enable_if<
    is_function<MFUNC>::value &&
    is_same_monad<M, typename MFUNC::result_type>::value &&
    std::is_same<value_type_t<M>, first_argument_type_t<MFUNC> >::value,
    mbind_result_type_t<M, MFUNC>
>::type;

template<typename M, typename MFUNC>
constexpr mbind_type<M, MFUNC> operator>>=(M const& m, MFUNC const& f){
    return Monad_t<M>::mbind(m, f);
}

template<typename M, typename MFUNC>
constexpr mbind_type<M, MFUNC> operator>>=(f0<M> const& mf, MFUNC const& f){
    return Monad_t<M>::mbind(*mf, f);
}

template<typename M, typename MFUNC>
constexpr mbind_type<M, MFUNC> operator<<=(MFUNC const& f, M const& m){
    return m >>= f;
}

template<typename M, typename MFUNC>
constexpr mbind_type<M, MFUNC> operator<<=(MFUNC const& f, f0<M> const& mf){
    return *mf >>= f;
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

template<typename M, typename Ret, typename Arg, typename... Args>
using liftM_type = typename std::enable_if<
    is_monad<M>::value && is_same_as<Arg, value_type_t<M> >::value,
    typename Monad_t<M>::template liftM_type<remove_f0_t<function_t<Ret(Args...)> > >
>::type;

#define LIFTM_TYPE_(M, Ret, Arg) BOOST_IDENTITY_TYPE((liftM_type<M, Ret, Arg, Args...>))
#define LIFTM_TYPE(M, Ret, Arg) typename LIFTM_TYPE_(M, Ret, Arg)

//template<typename M, typename Ret, typename Arg, typename... Args>
//liftM_type<M, Ret, Arg, Args...>
//liftM(function_t<Ret(Arg, Args...)> const&, M const&);

DECLARE_FUNCTION_2_ARGS(3, LIFTM_TYPE(T0, T1, T2), liftM, function_t<T1(T2, Args...)> const&, T0 const&);

//template<typename M, typename Ret, typename Arg, typename... Args>
//function_t<liftM_type<M, Ret, Arg, Args...>(M const&)>
//_liftM(function_t<Ret(Arg, Args...)> const& f){
//  return [f](M const& m){
//      return liftM(f, m);
//  };
//}

// liftM2 f m1 m2 = do { x1 <- m1; x2 <- m2; return (f x1 x2) }
template<typename M1, typename M2, typename Ret, typename Arg1, typename Arg2>
using liftM2_type = typename std::enable_if<
    is_same_monad<M1, M2>::value &&
    is_same_as<Arg1, value_type_t<M1> >::value &&
    is_same_as<Arg2, value_type_t<M2> >::value,
    typename Monad_t<M1>::template liftM_type<Ret>
>::type;

#define LIFTM2_TYPE_(M1, M2, Ret, Arg1, Arg2) BOOST_IDENTITY_TYPE((liftM2_type<M1, M2, Ret, Arg1, Arg2>))
#define LIFTM2_TYPE(M1, M2, Ret, Arg1, Arg2) typename LIFTM2_TYPE_(M1, M2, Ret, Arg1, Arg2)

DECLARE_FUNCTION_3(5, constexpr LIFTM2_TYPE(T0, T1, T2, T3, T4), liftM2, function_t<T2(T3, T4)> const&, T0 const&, T1 const&);

// liftM3 f m1 m2 m3 = do { x1 <- m1; x2 <- m2; x3 <- m3; return (f x1 x2 x3) }
template<typename M1, typename M2, typename M3, typename Ret, typename Arg1, typename Arg2, typename Arg3>
using liftM3_type = typename std::enable_if<
    is_same_monad<M1, M2>::value && is_same_monad<M1, M3>::value &&
    is_same_as<Arg1, value_type_t<M1> >::value &&
    is_same_as<Arg2, value_type_t<M2> >::value && is_same_as<Arg3, value_type_t<M3> >::value,
    typename Monad_t<M1>::template liftM_type<Ret>
>::type;

#define LIFTM3_TYPE_(M1, M2, M3, Ret, Arg1, Arg2, Arg3) BOOST_IDENTITY_TYPE((liftM3_type<M1, M2, M3, Ret, Arg1, Arg2, Arg3>))
#define LIFTM3_TYPE(M1, M2, M3, Ret, Arg1, Arg2, Arg3) typename LIFTM3_TYPE_(M1, M2, M3, Ret, Arg1, Arg2, Arg3)

DECLARE_FUNCTION_4(7, constexpr LIFTM3_TYPE(T0, T1, T2, T3, T4, T5, T6), liftM3, function_t<T3(T4, T5, T6)> const&, T0 const&, T1 const&, T2 const&);

// liftM4 f m1 m2 m3 m4 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; return (f x1 x2 x3 x4) }
template<typename M1, typename M2, typename M3, typename M4, typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
using liftM4_type = typename std::enable_if<
    is_same_monad<M1, M2>::value && is_same_monad<M1, M3>::value && is_same_monad<M1, M4>::value &&
    is_same_as<Arg1, value_type_t<M1> >::value &&
    is_same_as<Arg2, value_type_t<M2> >::value &&
    is_same_as<Arg3, value_type_t<M3> >::value &&
    is_same_as<Arg4, value_type_t<M4> >::value,
    typename Monad_t<M1>::template liftM_type<Ret>
>::type;

#define LIFTM4_TYPE_(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4) BOOST_IDENTITY_TYPE((liftM4_type<M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4>))
#define LIFTM4_TYPE(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4) typename LIFTM4_TYPE_(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4)

DECLARE_FUNCTION_5(9, constexpr LIFTM4_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8), liftM4,
    function_t<T4(T5, T6, T7, T8)> const&, T0 const&, T1 const&, T2 const&, T3 const&);

// liftM5 f m1 m2 m3 m4 m5 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; x5 <- m5; return (f x1 x2 x3 x4 x5) }
template<typename M1, typename M2, typename M3, typename M4, typename M5, typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
using liftM5_type = typename std::enable_if<
    is_same_monad<M1, M2>::value && is_same_monad<M1, M3>::value &&
    is_same_monad<M1, M4>::value && is_same_monad<M1, M5>::value &&
    is_same_as<Arg1, value_type_t<M1> >::value &&
    is_same_as<Arg2, value_type_t<M2> >::value &&
    is_same_as<Arg3, value_type_t<M3> >::value &&
    is_same_as<Arg4, value_type_t<M4> >::value &&
    is_same_as<Arg5, value_type_t<M5> >::value,
    typename Monad_t<M1>::template liftM_type<Ret>
>::type;

#define LIFTM5_TYPE_(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5) BOOST_IDENTITY_TYPE((liftM5_type<M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5>))
#define LIFTM5_TYPE(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5) typename LIFTM5_TYPE_(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5)

DECLARE_FUNCTION_6(11, constexpr LIFTM5_TYPE(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10), liftM5,
    function_t<T5(T6, T7, T8, T9, T10)> const&, T0 const&, T1 const&, T2 const&, T3 const&, T4 const&);

/*
 In many situations, the 'liftM' operations can be replaced by uses of
'ap', which promotes function application.

return f `ap` x1 `ap` ... `ap` xn
is equivalent to
liftMn f x1 x2 ... xn

ap                :: (Monad m) => m (a -> b) -> m a -> m b
ap m1 m2          = do { x1 <- m1; x2 <- m2; return (x1 x2) }
*/
template<typename MV, typename MF>
using ap_type = typename std::enable_if<
    is_same_monad<MF, MV>::value && is_function<value_type_t<MF> >::value &&
    is_same_as<first_argument_type_t<value_type_t<MF> >, value_type_t<MV> >::value,
    typeof_t<MF, remove_f0_t<remove_first_arg_t<value_type_t<MF> > > >
>::type;

#define AP_TYPE_(MV, MF) BOOST_IDENTITY_TYPE((ap_type<MV, MF>))
#define AP_TYPE(MV, MF) typename AP_TYPE_(MV, MF)

DECLARE_FUNCTION_2(2, constexpr AP_TYPE(T0, T1), ap, T1 const&, T0 const&);

template<typename MV, typename MF>
constexpr ap_type<MV, MF> operator&(MF const& mf, MV const& mv){
    return ap(mf, mv);
}

// mcompose
// f >=> g = \x -> (f x >>= g)
template<typename MG, typename ArgG, typename... ArgsG, typename MF, typename ArgF, typename... ArgsF>
constexpr typename std::enable_if<
    is_same_monad<MF, MG>::value && is_same_as<value_type_t<MF>, ArgG>::value,
    function_t<remove_f0_t<function_t<MG(ArgsG...)> >(ArgF, ArgsF...)>
>::type mcompose(function_t<MF(ArgF, ArgsF...)> const& f, function_t<MG(ArgG, ArgsG...)> const& g){
    return _([f, g](ArgF x, ArgsF... args){
        return f(x, args...) >>= g;
    });
}

template<typename MG, typename ArgG, typename... ArgsG, typename MF, typename ArgF, typename... ArgsF>
constexpr typename std::enable_if<
    is_same_monad<MF, MG>::value && is_same_as<value_type_t<MF>, ArgG>::value,
    function_t<function_t<remove_f0_t<function_t<MG(ArgsG...)> >(ArgF, ArgsF...)>
        (function_t<MG(ArgG, ArgsG...)> const&)>
>::type _mcompose(function_t<MF(ArgF, ArgsF...)> const& f){
    return [f](function_t<MG(ArgG, ArgsG...)> const& g){
        return mcompose(f, g);
    };
}

// do notation
#define _do(var, mexpr, expr) \
    ((mexpr) >>= _([=](typename std::decay<decltype(mexpr)>::type::value_type const& var){ expr }))

#define _do2(var1, mexpr1, var2, mexpr2, expr) \
    _do(var1, mexpr1, return _do(var2, mexpr2, expr);)

#define _do3(var1, mexpr1, var2, mexpr2, var3, mexpr3, expr) \
    _do2(var1, mexpr1, var2, mexpr2, return _do(var3, mexpr3, expr);)

#define _do4(var1, mexpr1, var2, mexpr2, var3, mexpr3, var4, mexpr4, expr) \
    _do3(var1, mexpr1, var2, mexpr2, var3, mexpr3, return _do(var4, mexpr4, expr);)

#define _do5(var1, mexpr1, var2, mexpr2, var3, mexpr3, var4, mexpr4, var5, mexpr5, expr) \
    _do4(var1, mexpr1, var2, mexpr2, var3, mexpr3, var4, mexpr4, return _do(var5, mexpr5, expr);)

_FUNCPROG_END

#include "detail/Monad_impl.hpp"
