#pragma once

#include "Applicative_fwd.hpp"

_FUNCPROG2_BEGIN

class monad_error : std::runtime_error
{
public:
    explicit monad_error(std::string const& _Message) : runtime_error(_Message){}
    explicit monad_error(const char* _Message) : runtime_error(_Message){}
};

template<class _M>
struct _is_monad : std::false_type {};

template<class _M>
bool constexpr _is_monad_v = _is_monad<_M>::value;

//template<class M>
//using is_monad = _is_monad<base_class_t<M> >;
template<class M>
struct is_monad : std::false_type {};

template<class M>
bool constexpr is_monad_v = is_monad<M>::value;

template<class M, typename T = M>
using monad_type = std::enable_if_t<is_monad_v<M>, T>;

#define MONAD_TYPE_(M, T) BOOST_IDENTITY_TYPE((monad_type<M, T>))
#define MONAD_TYPE(M, T) typename MONAD_TYPE_(M, T)

template<class _M1, class _M2>
struct _is_same_monad : std::false_type {};

template<class _M>
struct _is_same_monad<_M, _M> : _is_monad<_M> {};

template<class _M1, class _M2>
bool constexpr _is_same_monad_v = _is_same_monad<_M1, _M2>::value;

template<class M1, class M2>
using is_same_monad = _is_same_monad<base_class_t<M1>, base_class_t<M2> >;

template<class M1, class M2>
bool constexpr is_same_monad_v = is_same_monad<M1, M2>::value;

template<class M1, class M2, typename T>
using same_monad_type = std::enable_if_t<is_same_monad<M1, M2>::value, T>;

#define SAME_MONAD_TYPE_(M1, M2, T) BOOST_IDENTITY_TYPE((same_monad_type<M1, M2, T>))
#define SAME_MONAD_TYPE(M1, M2, T) typename SAME_MONAD_TYPE_(M1, M2, T)

// Requires mreturn and >>=
template<typename M>
struct Monad;

template<typename T>
using Monad_t = Monad<base_class_t<T> >;

#define IMPLEMENT_MONAD(_M, M) \
    template<> struct _is_monad<_M> : std::true_type {}; \
    template<typename A> struct is_monad<M<A> > : std::true_type {}

template<typename M>
using join_type = same_monad_type<M, typename M::value_type, value_type_t<M> >;

template<typename M>
constexpr join_type<M> join(M const& v){
    return v >>= id<typename M::value_type>();
}

// >=>
template<typename RetG, typename ArgG, typename FuncImplG, typename RetF, typename... ArgsF, typename FuncImplF>
constexpr auto operator^(function2<RetF(ArgsF...), FuncImplF> const& f, function2<RetG(ArgG const&), FuncImplG> const& g)
{
    static_assert(is_same_monad_v<RetF, RetG>, "Should be the same monads");
    static_assert(is_same_as_v<ArgG, typename RetF::value_type>, "Should be the same");
    return _([g, f](ArgsF... args){ return f(args...) >>= g; });
}

template<typename MFUNC>
using mbind_type1 = std::enable_if_t<
    is_function_v<MFUNC> &&
    is_monad_v<result_type_t<MFUNC> > &&
    !is_parser_v<result_type_t<MFUNC> >,
    remove_f0_t<remove_first_arg_t<MFUNC> >
>;

template<typename MFUNC>
using mbind_arg_type = typeof_t<result_type_t<MFUNC>, fdecay<first_argument_type_t<MFUNC> > >;

template<typename MFUNC>
constexpr auto operator>>=(mbind_arg_type<MFUNC> const& m, MFUNC const& f)
{
    static_assert(is_function_v<MFUNC>, "Should be a function");
    static_assert(is_monad_v<result_type_t<MFUNC> >, "Should be a monad");
    return Monad_t<result_type_t<MFUNC> >::mbind(m, f);
}

template<typename MFUNC, typename FuncImpl>
constexpr auto operator>>=(f0<mbind_arg_type<MFUNC>, FuncImpl> const& mf, MFUNC const& f){
    return *mf >>= f;
}

template<typename MFUNC>
constexpr auto operator<<=(MFUNC const& f, mbind_arg_type<MFUNC> const& m){
    return m >>= f;
}

template<typename MFUNC, typename FuncImpl>
constexpr auto operator<<=(MFUNC const& f, f0<mbind_arg_type<MFUNC>, FuncImpl> const& mf){
    return *mf >>= f;
}

template<typename MF, typename MG>
constexpr same_monad_type<MF, MG, MG> operator>>(MF const& f, MG const& g);

template<typename RetF, typename FuncImplF, typename RetG, typename FuncImplG>
constexpr same_monad_type<RetF, RetG, f0<RetG, FuncImplG> > operator>>(f0<RetF, FuncImplF> const& f, f0<RetG, FuncImplG> const& g);

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
using liftM_type = std::enable_if_t<
    is_monad_v<M> && is_same_as_v<Arg, value_type_t<M> >,
    typename Monad_t<M>::template liftM_type<remove_f0_t<function2<Ret(Args...), void> > >
>;

#define LIFTM_TYPE_(M, Ret, Arg) BOOST_IDENTITY_TYPE((liftM_type<M, Ret, Arg, Args...>))
#define LIFTM_TYPE(M, Ret, Arg) typename LIFTM_TYPE_(M, Ret, Arg)

//template<typename M, typename Ret, typename Arg, typename... Args>
//liftM_type<M, Ret, Arg, Args...>
//liftM(function2<Ret(Arg, Args...)> const&, M const&);

DECLARE_FUNCTION_2_ARGS(4, LIFTM_TYPE(T0, T2, T3), liftM, FUNCTION2(T2(T3, Args...), T1) const&, T0 const&);

// liftM2 f m1 m2 = do { x1 <- m1; x2 <- m2; return (f x1 x2) }
template<typename M1, typename M2, typename Ret, typename Arg1, typename Arg2>
using liftM2_type = std::enable_if_t<
    is_same_monad_v<M1, M2> &&
    is_same_as_v<Arg1, value_type_t<M1> > &&
    is_same_as_v<Arg2, value_type_t<M2> >,
    typename Monad_t<M1>::template liftM_type<Ret>
>;

#define LIFTM2_TYPE_(M1, M2, Ret, Arg1, Arg2) BOOST_IDENTITY_TYPE((liftM2_type<M1, M2, Ret, Arg1, Arg2>))
#define LIFTM2_TYPE(M1, M2, Ret, Arg1, Arg2) typename LIFTM2_TYPE_(M1, M2, Ret, Arg1, Arg2)

DECLARE_FUNCTION_3(6, LIFTM2_TYPE(T0, T1, T3, T4, T5), liftM2, FUNCTION2(T3(T4, T5), T2) const&, T0 const&, T1 const&);

// liftM3 f m1 m2 m3 = do { x1 <- m1; x2 <- m2; x3 <- m3; return (f x1 x2 x3) }
template<typename M1, typename M2, typename M3, typename Ret, typename Arg1, typename Arg2, typename Arg3>
using liftM3_type = std::enable_if_t<
    is_same_monad_v<M1, M2> && is_same_monad_v<M1, M3> &&
    is_same_as_v<Arg1, value_type_t<M1> > &&
    is_same_as_v<Arg2, value_type_t<M2> > &&
    is_same_as_v<Arg3, value_type_t<M3> >,
    typename Monad_t<M1>::template liftM_type<Ret>
>;

#define LIFTM3_TYPE_(M1, M2, M3, Ret, Arg1, Arg2, Arg3) BOOST_IDENTITY_TYPE((liftM3_type<M1, M2, M3, Ret, Arg1, Arg2, Arg3>))
#define LIFTM3_TYPE(M1, M2, M3, Ret, Arg1, Arg2, Arg3) typename LIFTM3_TYPE_(M1, M2, M3, Ret, Arg1, Arg2, Arg3)

DECLARE_FUNCTION_4(8, LIFTM3_TYPE(T0, T1, T2, T4, T5, T6, T7), liftM3, FUNCTION2(T4(T5, T6, T7), T3) const&, T0 const&, T1 const&, T2 const&);

// liftM4 f m1 m2 m3 m4 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; return (f x1 x2 x3 x4) }
template<typename M1, typename M2, typename M3, typename M4, typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
using liftM4_type = std::enable_if_t<
    is_same_monad_v<M1, M2> && is_same_monad_v<M1, M3> && is_same_monad_v<M1, M4> &&
    is_same_as_v<Arg1, value_type_t<M1> > &&
    is_same_as_v<Arg2, value_type_t<M2> > &&
    is_same_as_v<Arg3, value_type_t<M3> > &&
    is_same_as_v<Arg4, value_type_t<M4> >,
    typename Monad_t<M1>::template liftM_type<Ret>
>;

#define LIFTM4_TYPE_(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4) BOOST_IDENTITY_TYPE((liftM4_type<M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4>))
#define LIFTM4_TYPE(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4) typename LIFTM4_TYPE_(M1, M2, M3, M4, Ret, Arg1, Arg2, Arg3, Arg4)

DECLARE_FUNCTION_5(10, LIFTM4_TYPE(T0, T1, T2, T3, T5, T6, T7, T8, T9), liftM4,
    FUNCTION2(T5(T6, T7, T8, T9), T4) const&, T0 const&, T1 const&, T2 const&, T3 const&);

// liftM5 f m1 m2 m3 m4 m5 = do { x1 <- m1; x2 <- m2; x3 <- m3; x4 <- m4; x5 <- m5; return (f x1 x2 x3 x4 x5) }
template<typename M1, typename M2, typename M3, typename M4, typename M5, typename Ret, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
using liftM5_type = std::enable_if_t<
    is_same_monad_v<M1, M2> && is_same_monad_v<M1, M3> &&
    is_same_monad_v<M1, M4> && is_same_monad_v<M1, M5> &&
    is_same_as_v<Arg1, value_type_t<M1> > &&
    is_same_as_v<Arg2, value_type_t<M2> > &&
    is_same_as_v<Arg3, value_type_t<M3> > &&
    is_same_as_v<Arg4, value_type_t<M4> > &&
    is_same_as_v<Arg5, value_type_t<M5> >,
    typename Monad_t<M1>::template liftM_type<Ret>
>;

#define LIFTM5_TYPE_(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5) BOOST_IDENTITY_TYPE((liftM5_type<M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5>))
#define LIFTM5_TYPE(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5) typename LIFTM5_TYPE_(M1, M2, M3, M4, M5, Ret, Arg1, Arg2, Arg3, Arg4, Arg5)

DECLARE_FUNCTION_6(12, LIFTM5_TYPE(T0, T1, T2, T3, T4, T6, T7, T8, T9, T10, T11), liftM5,
    FUNCTION2(T6(T7, T8, T9, T10, T11), T5) const&, T0 const&, T1 const&, T2 const&, T3 const&, T4 const&);

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
using ap_type = std::enable_if_t<
    is_same_monad<MF, MV>::value&& is_function_v<value_type_t<MF> > &&
    is_same_as_v<first_argument_type_t<value_type_t<MF> >, value_type_t<MV> >,
    typeof_dt<MF, remove_first_arg_t<value_type_t<MF> > >
>;

#define AP_TYPE_(MV, MF) BOOST_IDENTITY_TYPE((ap_type<MV, MF>))
#define AP_TYPE(MV, MF) typename AP_TYPE_(MV, MF)

DECLARE_FUNCTION_2(2, AP_TYPE(T0, T1), ap, T1 const&, T0 const&);

template<typename MV, typename MF>
constexpr ap_type<MV, MF> operator&(MF const& mf, MV const& mv){
    return ap(mf, mv);
}

// mcompose
// f >=> g = \x -> (f x >>= g)
template<typename FuncImplG, typename MG, typename ArgG, typename... ArgsG, typename FuncImplF, typename MF, typename ArgF, typename... ArgsF>
constexpr auto mcompose(function2<MF(ArgF, ArgsF...), FuncImplF> const& f, function2<MG(ArgG, ArgsG...), FuncImplG> const& g)
{
    static_assert(is_same_monad_v<MF, MG> && is_same_as_v<value_type_t<MF>, ArgG>, "Should be the same monad");
    return _([f, g](ArgF x, ArgsF... args){
        return f(x, args...) >>= g;
    });
}

template<typename FuncImplG, typename MG, typename ArgG, typename... ArgsG, typename FuncImplF, typename MF, typename ArgF, typename... ArgsF>
constexpr auto _mcompose(function2<MF(ArgF, ArgsF...), FuncImplF> const& f)
{
    static_assert(is_same_monad_v<MF, MG> && is_same_as_v<value_type_t<MF>, ArgG>, "Should be the same monad");
    return _([f](function2<MG(ArgG, ArgsG...), FuncImplG> const& g){
        return mcompose(f, g);
    });
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

_FUNCPROG2_END
