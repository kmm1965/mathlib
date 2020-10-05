#pragma once

#include "../../Identity.hpp"
#include "../../Alternative.hpp"
#include "MonadTrans.hpp"
#include "../MonadReader.hpp"
#include "../MonadWriter.hpp"
#include "../MonadState.hpp"

_FUNCPROG_BEGIN

template<class _F>
struct _IdentityT;

template<class _F, typename A>
struct IdentityT;

#define IDENTITYT_(_F, A) BOOST_IDENTITY_TYPE((IdentityT<_F, A>))
#define IDENTITYT(_F, A) typename IDENTITYT_(_F, A)

struct __IdentityT
{
    template<class _F>
    using mt_type = _IdentityT<_F>;

    // -- | Lift a unary operation to the new monad.
    // mapIdentityT :: (m a -> n b) -> IdentityT m a -> IdentityT n b
    // mapIdentityT f = IdentityT . f . runIdentityT
    template<typename MA, typename NB>
    static IdentityT<base_class_t<NB>, value_type_t<NB> >
    map(function_t<NB(MA const&)> const& f, IdentityT<base_class_t<MA>, value_type_t<MA> > const& m) {
        return f(m.run());
    }

};

template<class _F>
struct _IdentityT
{
    static_assert(is_monad<_F>::value, "Should be a monad");

    using base_class = _IdentityT;

    template<typename A>
    using type = IdentityT<_F, A>;
};

template<class _F, typename A>
struct IdentityT : _IdentityT<_F>
{
    static_assert(!is_function0<A>::value, "Should not be a function");

    using super = _IdentityT<_F>;
    using value_type = A;
    using value_t = typename _F::template type<A>;

    IdentityT(value_t const& value) : value(value) {}
    IdentityT(value_type const& value) : value(value) {}

    value_t const& run() const {
        return value;
    }

private:
    const value_t value;
};

template<typename FA>
IdentityT<base_class_t<FA>, value_type_t<FA> > IdentityT_(FA const& value) {
    return value;
}

template<class _F, typename A>
typename _F::template type<A> runIdentityT(IdentityT<_F, A> const& x) {
    return x.run();
}

template<typename MA, typename NB>
IdentityT<base_class_t<NB>, value_type_t<NB> >
mapIdentityT(function_t<NB(MA const&)> const& f, IdentityT<base_class_t<MA>, value_type_t<MA> > const& m){
    return __IdentityT::map(f, m);
}

template<typename MA, typename NB>
function_t<IdentityT<base_class_t<NB>, value_type_t<NB> >(
    IdentityT<base_class_t<MA>, value_type_t<MA> > const&
)> _mapIdentityT(function_t<NB(MA const&)> const& f) {
    return [f](IdentityT<base_class_t<MA>, value_type_t<MA> > const& m) {
        return mapIdentityT(f, m);
    };
}

// -- | Lift a binary operation to the new monad.
// lift2IdentityT ::
//  (m a -> n b -> p c) -> IdentityT m a -> IdentityT n b -> IdentityT p c
// lift2IdentityT f a b = IdentityT (f (runIdentityT a) (runIdentityT b))
template<typename MA, typename NB, typename PC>
static IdentityT<base_class_t<PC>, value_type_t<PC> >
lift2IdentityT(function_t<PC(MA const&, NB const&)> const& f,
    IdentityT<base_class_t<MA>, value_type_t<MA> > const& m,
    IdentityT<base_class_t<NB>, value_type_t<NB> > const& n)
{
    return f(m.run(), n.run());
}

template<typename T>
struct is_IdentityT : std::false_type {};

template<class _F, typename A>
struct is_IdentityT<IdentityT<_F, A> > : std::true_type {};

// Functor
template<class _F>
struct _is_functor<_IdentityT<_F> > : _is_functor<_F> {};

template<class _F>
struct Functor<_IdentityT<_F> >
{
    using base_class = _IdentityT<_F>;

    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    // fmap :: (a -> b) -> IdentityT a -> IdentityT b
    // fmap f = mapIdentityT (fmap f)
    template<typename Ret, typename Arg, typename... Args>
    static IdentityT<_F, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, IdentityT<_F, fdecay<Arg> > const& v)  {
        return mapIdentityT(_fmap<typename _F::template type<fdecay<Arg> > >(f), v);
    }
};

// Applicative
template<class _F>
struct _is_applicative<_IdentityT<_F> > : _is_applicative<_F> {};

template<class _F>
struct Applicative<_IdentityT<_F> > : Functor<_IdentityT<_F> >
{
    using base_class = _IdentityT<_F>;
    using super = Functor<base_class>;

    template<typename A>
    static IdentityT<_F, A> pure(A const& x) {
        return Applicative<_F>::pure(x);
    }

    // (<*>) = lift2IdentityT (<*>)
    template<typename Ret, typename Arg, typename... Args>
    static IdentityT<_F, remove_f0_t<function_t<Ret(Args...)> > >
    apply(IdentityT<_F, function_t<Ret(Arg, Args...)> > const& f, IdentityT<_F, fdecay<Arg> > const& v){
        return lift2IdentityT(_(Applicative<_F>::template apply<Ret, Arg, Args...>), f, v);
    }
};

template<class _F, typename Fa, typename Fb>
typename std::enable_if<is_same_applicative<Fa, Fb>::value, IdentityT<_F, Fb> >::type
operator*=(IdentityT<_F, Fa> const& a, IdentityT<_F, Fb> const& b) {
    return lift2IdentityT<_F>(_(ap_r<Fa, Fb>), a, b);
}

template<class _F, typename Fa, typename Fb>
typename std::enable_if<is_same_applicative<Fa, Fb>::value, IdentityT<_F, Fa> >::type
operator^=(IdentityT<_F, Fa> const& a, IdentityT<_F, Fb> const& b) {
    return lift2IdentityT<_F>(_(ap_l<Fa, Fb>), a, b);
}

// Monad
template<class _F>
struct _is_monad<_IdentityT<_F> > : _is_monad<_F> {};

template<class _F>
struct Monad<_IdentityT<_F> > : Applicative<_IdentityT<_F> >
{
    using base_class = _IdentityT<_F>;
    using super = Applicative<base_class>;

    template<typename A>
    using liftM_type = IdentityT<_F, A>;

    // return = IdentityT . return
    template<typename A>
    static IdentityT<_F, A> mreturn(A const& x) {
        return Monad<_F>::mreturn(x);
    }

    // fail msg = IdentityT $ fail msg
    static IdentityT<_F, const char*> fail(const char* msg) {
        return Monad<_F>::fail(msg);
    }

    // m >>= k = IdentityT $ runIdentityT . k =<< runIdentityT m
    template<typename Ret, typename Arg, typename... Args>
    static remove_f0_t<function_t<IdentityT<_F, Ret>(Args...)> >
    mbind(IdentityT<_F, fdecay<Arg> > const& m, function_t<IdentityT<_F, Ret>(Arg, Args...)> const& f){
        return (_(runIdentityT<_F, Ret>) & f) <<= m.run();
    }
};

// MonadPlus
template<class _F>
struct is_monad_plus<_IdentityT<_F> > : is_monad_plus<_F> {};

template<class _F>
struct is_same_monad_plus<_IdentityT<_F>, _IdentityT<_F> > : is_monad_plus<_F> {};

template<class _F>
struct MonadPlus<_IdentityT<_F> > : Monad<_IdentityT<_F> >
{
    using base_class = _IdentityT<_F>;
    using super = Monad<base_class>;

    template<typename A>
    static IdentityT<_F, A> mzero() {
        return MonadPlus<_F>::template mzero<A>();
    }

    template<typename T>
    struct mplus_result_type;

    template<typename T>
    using mplus_result_type_t = typename mplus_result_type<T>::type;
    
    template<typename A>
    struct mplus_result_type<IdentityT<_F, A> >
    {
        using type = IdentityT<_F, A>;
    };
    
    template<typename A>
    static IdentityT<_F, A> mplus(IdentityT<_F, A> const& l, IdentityT<_F, A> const& r){
        return lift2IdentityT(_(MonadPlus<_F>::template mplus<A>), l, r);
    }
};

// Alternative
template<class _F>
struct is_alternative<_IdentityT<_F> > : is_alternative<_F> {};

template<class _F>
struct is_same_alternative<_IdentityT<_F>, _IdentityT<_F> > : is_alternative<_F> {};

template<class _F>
struct Alternative<_IdentityT<_F> >
{
    using base_class = _IdentityT<_F>;

    template<typename A>
    static IdentityT<_F, A> empty() {
        return Alternative<_F>::template empty<A>();
    }

    template<typename T>
    struct alt_op_result_type;

    template<typename T>
    using alt_op_result_type_t = typename alt_op_result_type<T>::type;
    
    template<typename A>
    struct alt_op_result_type<IdentityT<_F, A> >
    {
        using type = IdentityT<_F, A>;
    };

    template<typename A>
    static IdentityT<_F, A> alt_op(IdentityT<_F, A> const& l, IdentityT<_F, A> const& r){
        return lift2IdentityT(_(Alternative<_F>::template alt_op<A>), l, r);
    }
};

// Foldable
template<class _F, typename A>
struct is_foldable<IdentityT<_F, A> > : is_foldable<typename _F::template type<A> > {};

template<class _F>
struct Foldable<_IdentityT<_F> >
{
    // foldMap :: Monoid m => (a -> m) -> t a -> m
    // foldMap f (IdentityT t) = foldMap f t
    template<typename M, typename Arg>
    static typename std::enable_if<is_monoid<M>::value, M>::type
    foldMap(function_t<M(Arg)> const& f, IdentityT<_F, fdecay<Arg> > const& x){
        return Foldable<_F>::foldMap(f, x.run());
    }

    // foldl :: (b -> a -> b) -> b -> t a -> b
    // foldl f z (IdentityT t) = foldl f z t
    template<typename B, typename Arg, typename BArg>
    static B foldl(function_t<B(BArg, Arg)> const& f, B const& z, IdentityT<_F, fdecay<Arg> > const& x)
    {
        static_assert(is_same_as<B, BArg>::value, "Should be the same");
        return Foldable<_F>::foldl(f, z, x.run());
    }

    // foldl1 :: (a -> a -> a) -> t a -> a
    // foldl1 f (IdentityT t) = foldl1 f t
    template<typename A, typename Arg1, typename Arg2>
    static A foldl1(function_t<A(Arg1, Arg2)> const& f, IdentityT<_F, A> const& x)
    {
        static_assert(is_same_as<A, Arg1>::value, "Should be the same");
        static_assert(is_same_as<A, Arg2>::value, "Should be the same");
        return Foldable<_F>::foldl1(f, x.run());
    }

    // foldr :: (a -> b -> b) -> b -> t a -> b
    // foldr f z (IdentityT t) = foldr f z t
    template<typename B, typename Arg, typename BArg>
    static B foldr(function_t<B(Arg, BArg)> const& f, B const& z, IdentityT<_F, fdecay<Arg> > const& x)
    {
        static_assert(is_same_as<B, BArg>::value, "Should be the same");
        return Foldable<_F>::foldr(f, z, x.run());
    }

    // foldr1 :: (a -> a -> a) -> t a -> a
    // foldr1 f (IdentityT t) = foldr1 f t
    template<typename A, typename Arg1, typename Arg2>
    static A foldr1(function_t<A(Arg1, Arg2)> const& f, IdentityT<_F, A> const& x)
    {
        static_assert(is_same_as<A, Arg1>::value, "Should be the same");
        static_assert(is_same_as<A, Arg2>::value, "Should be the same");
        return Foldable<_F>::foldr1(f, x.run());
    }
};

// Traversable
template<class _F, typename A>
struct is_traversable<IdentityT<_F, A> > : is_traversable<typename _F::template type<A> > {};

template<class _F>
struct Traversable<_IdentityT<_F> >
{
    template<typename A>
    using F_type = typename _F::template type<A>;

    // traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
    // traverse f (IdentityT a) = IdentityT <$> traverse f a
    template<typename AP, typename Arg>
    static typename std::enable_if<is_applicative<AP>::value, typeof_t<AP, IdentityT<_F, value_type_t<AP> > > >::type
    traverse(function_t<AP(Arg)> const& f, IdentityT<_F, fdecay<Arg> > const& x){
        return _(IdentityT_<F_type<value_type_t<AP> > >) / Traversable<_F>::traverse(f, x.run());
    }

    // sequenceA :: Applicative f => t (f a) -> f (t a)
    template<typename A>
    static F_type<IdentityT<_F, A> >
    sequenceA(IdentityT<_F, A> const& x)
    {
        static_assert(is_applicative<_F>::value, "Should be an Applicative");
        return traverse(_(id<F_type<A> >), x);
    }
};

// MonadZip
template<class _F>
struct MonadZip<_IdentityT<_F> > : _MonadZip<MonadZip<_IdentityT<_F> > >
{
    using base_class = _IdentityT<_F>;

    template<typename A>
    using F_type = typename _F::template type<fdecay<A> >;

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f = lift2IdentityT (mzipWith f)
    template<typename C, typename A, typename B>
    static IdentityT<_F, C> mzipWith(function_t<C(A, B)> const& f,
        IdentityT<_F, fdecay<A> > const& ma, IdentityT<_F, fdecay<B> > const& mb)
    {
        return lift2IdentityT(_mzipWith<F_type<B>, F_type<A> >(f), ma, mb);
    }
};

// MonadTrans
template<class _F>
struct MonadTrans<_IdentityT<_F> >
{
    // lift = IdentityT
    template<typename A>
    static IdentityT<_F, A> lift(typename _F::template type<A> const& m){
        return m;
    }
};

// MonadReader
template<typename R, class _F>
struct MonadReader<R, _IdentityT<_F> > : _MonadReader<R, _IdentityT<_F>, MonadReader<R, _IdentityT<_F> > >
{
    using base_class = _IdentityT<_F>;
    using super = _MonadReader<R, base_class, MonadReader<R, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // ask = lift ask
    static type<R> ask() {
        return MonadTrans<base_class>::lift(MonadReader<R, _F>::ask());
    }

    // local = mapIdentityT . local
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    local(function_t<R(RArg)> const& f, type<A> const& m)
    {
        using FA = typename _F::template type<A>;
        return (_(mapIdentityT<FA, FA>) & _(MonadReader<R, _F>::template local<A, RArg>))(f, m);
    }

    // reader = lift . reader
    template<typename A, typename RArg>
    static typename std::enable_if< is_same_as<R, RArg>::value, type<A> >::type
    reader(function_t<A(RArg)> const& f){
        return MonadTrans<base_class>::lift(MonadReader<R, _F>::reader(f));
    }
};

// MonadWriter
template<typename W, class _F>
struct MonadWriter<W, _IdentityT<_F> > : _MonadWriter<W, _IdentityT<_F>, MonadWriter<W, _IdentityT<_F> > >
{
    using base_class = _IdentityT<_F>;
    using super = _MonadWriter<W, base_class, MonadWriter<W, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // writer = lift . writer
    template<typename A>
    static type<A> writer(pair_t<A, W> const& p) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _F>::writer(p));
    }

    // tell = lift . tell
    static type<None> tell(W const& w) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _F>::tell(w));
    }

    // listen = mapIdentityT listen
    template<typename A>
    static type<pair_t<A, W> > listen(IdentityT<_F, A> const& m) {
        return mapIdentityT(_(MonadWriter<W, _F>::template listen<A>), m);
    }

    // pass = mapIdentityT pass 
    template<typename A>
    static type<A> pass(IdentityT<_F, pair_t<A, function_t<W(W const&)> > > const& m) {
        return mapIdentityT(_(MonadWriter<W, _F>::template pass<A>), m);
    }
};

// MonadState
template<typename S, typename _M>
struct MonadState<S, _IdentityT<_M> > : _MonadState<S, _IdentityT<_M>, MonadState<S, _IdentityT<_M> > >
{
    using base_class = _IdentityT<_M>;
    using super = _MonadState<S, base_class, MonadState>;

    template<typename T>
    using type = typename super::template type<T>;

    // state = lift . state
    template<typename A>
    static type<A> state(function_t<pair_t<A, S>(S const&)> const& f) {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::state(f));
    }

    // get = lift get
    static type<S> get() {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::get());
    }

    // put = lift . put
    static type<None> put(S const& s) {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::put(s));
    }
};

_FUNCPROG_END

namespace std {

    template<class _F, typename A>
    ostream& operator<<(ostream& os, _FUNCPROG::IdentityT<_F, A> const& v) {
        return os << "IdentityT[" << v.run() << ']';
    }

}
