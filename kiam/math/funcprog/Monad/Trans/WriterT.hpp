#pragma once

#include "../../Identity.hpp"
#include "../../Foldable.hpp"
#include "../../Traversable.hpp"
#include "../MonadReader.hpp"
#include "../MonadWriter.hpp"
#include "../MonadState.hpp"

_FUNCPROG_BEGIN

/*
-- ---------------------------------------------------------------------------
-- | A writer monad parameterized by the type @w@ of output to accumulate.
--
-- The 'return' function produces the output 'mempty', while @>>=@
-- combines the outputs of the subcomputations using 'mappend'.
*/
template<typename W, class _M>
struct _WriterT;

template<typename W>
using _Writer = _WriterT<W, _Identity>;

template<typename W, class _M, typename A>
struct WriterT;

#define WRITERT_(W, _M, A) BOOST_IDENTITY_TYPE((WriterT<W, _M, A>))
#define WRITERT(W, _M, A) typename WRITERT_(W, _M, A)

template<typename W, typename A>
using Writer = WriterT<W, _Identity, A>;

#define WRITER_(W, A) BOOST_IDENTITY_TYPE((Writer<W, A>))
#define WRITER(W, A) typename WRITER_(W, A)

/*
-- | Construct a writer computation from a (result, output) pair.
-- (The inverse of 'runWriter'.)
writer :: (Monad m) => (a, w) -> WriterT w m a
writer = WriterT . return
*/
template<class _M, typename W, typename A>
WriterT<W, _M, A> writer(pair_t<A, W> const& value){
    return Monad<_M>::mreturn(value);
}

template<typename W>
struct __WriterT
{
    template<class _M>
    using base_type = _WriterT<W, _M>;
};

template<typename W, class _M>
struct _WriterT : __WriterT<W>
{
    static_assert(_is_monad_v<_M>, "Should be a monad");

    using base_class = _WriterT;

    template<typename A>
    using type = WriterT<W, _M, A>;

    // -- | @'tell' w@ is an action that produces the output @w@.
    // tell :: (Monad m) => w -> WriterT w m ()
    // tell w = writer ((), w)
    static constexpr WriterT<W, _M, None> tell(W const& w){
        return writer<_M>(pair_t<None, W>(None(), w));
    }

    // -- | @'pass' m@ is an action that executes the action @m@, which returns
    // -- a value and a function, and returns the value, applying the function
    // -- to the output.
    // --
    // -- * @'runWriterT' ('pass' m) = 'liftM' (\\ ((a, f), w) -> (a, f w)) ('runWriterT' m)@
    // pass :: (Monad m) => WriterT w m (a, w -> w) -> WriterT w m a
    // pass m = WriterT $ do
    //   ~((a, f), w) <- runWriterT m
    //   return (a, f w)
    template<typename A>
    static constexpr WriterT<W, _M, A> pass(WriterT<W, _M, pair_t<A, function_t<W(W const&)> > > const& m){
        return _do(pf, m.run(), return Monad<_M>::mreturn(pair_t<A, W>(fst(fst(pf)), snd(fst(pf))(snd(pf)))););
    }
};

/*
-- ---------------------------------------------------------------------------
-- | A writer monad parameterized by:
--
--   * @w@ - the output to accumulate.
--
--   * @m@ - The inner monad.
--
-- The 'return' function produces the output 'mempty', while @>>=@
-- combines the outputs of the subcomputations using 'mappend'.
*/
template<typename W, class _M, typename A>
struct WriterT : _WriterT<W, _M>
{
    using value_type = A;
    using value_t = typename _M::template type<pair_t<A, W> >;

    WriterT(value_t const& value) : value(value){}

    value_t const& run() const {
        return value;
    }

    /*
    -- | Extract the output from a writer computation.
    --
    -- * @'execWriterT' m = 'liftM' 'snd' ('runWriterT' m)@
    execWriterT :: (Monad m) => WriterT w m a -> m w
    execWriterT m = do
        ~(_, w) <- runWriterT m
        return w
    */
    typename _M::template type<W> exec() const {
        return _do(p, run(), return Monad<_M>::mreturn(snd(p)););
    }

    // -- | @'listen' m@ is an action that executes the action @m@ and adds its
    // -- output to the value of the computation.
    // --
    // -- * @'runWriterT' ('listen' m) = 'liftM' (\\ (a, w) -> ((a, w), w)) ('runWriterT' m)@
    // listen :: (Monad m) => WriterT w m a -> WriterT w m (a, w)
    // listen m = WriterT $ do
    //   ~(a, w) <- runWriterT m
    //   return ((a, w), w)
    WriterT<W, _M, pair_t<A, W> > listen() const {
        return _do(pa, run(),
            return Monad<_M>::mreturn(make_pair_t(make_pair_t(fst(pa), snd(pa)), snd(pa))););
    }

    // -- | @'listens' f m@ is an action that executes the action @m@ and adds
    // -- the result of applying @f@ to the output to the value of the computation.
    // --
    // -- * @'listens' f m = 'liftM' (id *** f) ('listen' m)@
    // --
    // -- * @'runWriterT' ('listens' f m) = 'liftM' (\\ (a, w) -> ((a, f w), w)) ('runWriterT' m)@
    // listens :: (Monad m) => (w -> b) -> WriterT w m a -> WriterT w m (a, b)
    // listens f m = WriterT $ do
    //   ~(a, w) <- runWriterT m
    //   return ((a, f w), w)
    template<typename B, typename WP>
    std::enable_if_t<is_same_as_v<W, WP>, WriterT<W, _M, pair_t<A, B> > >
    listens(function_t<B(WP)> const& f) const {
        return _do(pa, run(),
            return Monad<_M>::mreturn(make_pair_t(make_pair_t(fst(pa), f(snd(pa))), snd(pa))););
    }

    // -- | @'censor' f m@ is an action that executes the action @m@ and
    // -- applies the function @f@ to its output, leaving the return value
    // -- unchanged.
    // --
    // -- * @'censor' f m = 'pass' ('liftM' (\\ x -> (x,f)) m)@
    // --
    // -- * @'runWriterT' ('censor' f m) = 'liftM' (\\ (a, w) -> (a, f w)) ('runWriterT' m)@
    // censor :: (Monad m) => (w -> w) -> WriterT w m a -> WriterT w m a
    // censor f m = WriterT $ do
    //   ~(a, w) <- runWriterT m
    //   return (a, f w)
    WriterT<W, _M, A> censor(function_t<W(W const&)> const& f) const {
        return _do(pa, run(),
            return Monad<_M>::mreturn(make_pair_t(fst(pa), f(snd(pa)))););
    }

private:
    const value_t value;
};

template<typename W, class _M, typename A>
WriterT<W, _M, A> WriterT_(typename _M::template type<pair_t<A, W> > const& value){
    return value;
}

template<typename W, class _M, typename A>
std::enable_if_t<_is_monad_v<_M>, typename _M::template type<pair_t<A, W> > >
runWriterT(WriterT<W, _M, A> const& m){
    return m.run();
}

template<typename W, class _M, typename A>
typename _M::template type<W> execWriterT(WriterT<W, _M, A> const& m){
    return m.exec();
}

/*
-- | Map both the return value and output of a computation using
-- the given function.
--
-- * @'runWriterT' ('mapWriterT' f m) = f ('runWriterT' m)@
mapWriterT :: (m (a, w) -> n (b, w')) -> WriterT w m a -> WriterT w' n b
mapWriterT f m = WriterT $ f (runWriterT m)
*/
template<typename MA, typename NB>
WriterT<snd_type_t<value_type_t<NB> >, base_class_t<NB>, fst_type_t<value_type_t<NB> > >
mapWriterT(function_t<NB(MA const&)> const& f, WriterT<snd_type_t<value_type_t<MA> >, base_class_t<MA>, fst_type_t<value_type_t<MA> > > const& m){
    return f(m.run());
}

template<typename W, typename MA, typename NB>
function_t<WriterT<snd_type_t<value_type_t<NB> >, base_class_t<NB>, value_type_t<NB> >(WriterT<snd_type_t<value_type_t<MA> >, base_class_t<MA>, value_type_t<MA> > const&)>
_mapWriterT(function_t<NB(MA const&)> const& f){
    return [f](WriterT<snd_type_t<value_type_t<MA> >, base_class_t<MA>, value_type_t<MA> > const& m){
        return mapWriterT(f, m);
    };
}

/*
-- | Unwrap a writer computation as a (result, output) pair.
-- (The inverse of 'writer'.)
runWriter :: Writer w a -> (a, w)
runWriter = runIdentity . runWriterT
*/
template<typename W, typename A>
pair_t<A, W> runWriter(Writer<W, A> const& m){
    return m.run().run();
}

template<typename W, typename A>
W execWriter(Writer<W, A> const& m){
    return snd(runWriter(m));
}

template<typename T>
struct is_writer : std::false_type {};

template<typename W, class _M, typename A>
struct is_writer<WriterT<W, _M, A> > : std::true_type {};

// Functor
template<typename W, class _M>
struct _is_functor<_WriterT<W, _M> > : _is_functor<_M> {};

template<typename W, class _M, typename A>
struct is_functor<WriterT<W, _M, A> > : _is_functor<_M> {};

template<typename W, class _M>
struct Functor<_WriterT<W, _M> > : _Functor<_WriterT<W, _M> >
{
    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    // fmap f = mapWriterT $ fmap $ \ ~(a, w) -> (f a, w)
    template<typename Ret, typename Arg, typename... Args>
    static constexpr WriterT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, WriterT<W, _M, fdecay<Arg> > const& v){
        using A = fdecay<Arg>;
        return mapWriterT(_fmap<typename _M::template type<pair_t<A, W> > >(_([f](pair_t<A, W> const& p){
            return pair_t<remove_f0_t<function_t<Ret(Args...)> >, W>(invoke_f0(f << fst(p)), snd(p));
        })), v);
    }
};

// Applicative
template<typename W, class _M>
struct _is_applicative<_WriterT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> && _is_applicative_v<_M> >{};

template<typename W, class _M, typename A>
struct is_applicative<WriterT<W, _M, A> > : std::integral_constant<bool, is_monoid_v<W>&& _is_applicative_v<_M> > {};

template<typename W, class _M>
struct Applicative<_WriterT<W, _M> > : Functor<_WriterT<W, _M> >, _Applicative<_WriterT<W, _M> >
{
    static_assert(is_monoid_v<W>, "Should be a Monoid");

    using super = Functor<_WriterT<W, _M>>;

    // pure a  = WriterT $ pure (a, mempty)
    template<typename A>
    static constexpr WriterT<W, _M, A> pure(A const& a){
        return Applicative<_M>::pure(make_pair_t(a, W::mempty()));
    }

    // f <*> v = WriterT $ liftA2 k (runWriterT f) (runWriterT v)
    //   where k ~(a, w) ~(b, w') = (a b, w `mappend` w')
    template<typename Ret, typename Arg, typename... Args>
    static constexpr WriterT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(WriterT<W, _M, function_t<Ret(Arg, Args...)> > const& f, WriterT<W, _M, fdecay<Arg> > const& v){
        return liftA2(_([](pair_t<function_t<Ret(Arg, Args...)>, W> const& p, pair_t<fdecay<Arg>, W> const& q){
            auto const& [a, w] = p;
            auto const& [b, w_] = q;
            return make_pair_t(a(b), Monoid_t<W>::mappend(w, w_));
        }), f.run(), v.run());
    }
};

// Monad
template<typename W, class _M>
struct _is_monad<_WriterT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> && _is_monad_v<_M> > {};

template<typename W, class _M, typename A>
struct is_monad<WriterT<W, _M, A> > : std::integral_constant<bool, is_monoid_v<W>&& _is_monad_v<_M> > {};

template<typename W, class _M>
struct Monad<_WriterT<W, _M> > : Applicative<_WriterT<W, _M> >, _Monad<_WriterT<W, _M> >
{
    using super = Applicative<_WriterT<W, _M> >;

    template<typename T>
    using liftM_type = WriterT<W, _M, T>;

    // return a = writer (a, mempty)
    template<typename A>
    static constexpr WriterT<W, _M, A> mreturn(A const& x){
        return writer<_M>(make_pair_t(x, W::mempty()));
    }

    // m >>= k  = WriterT $ do
    //   ~(a, w)  <- runWriterT m
    //   ~(b, w') <- runWriterT (k a)
    //   return (b, w `mappend` w')
    template<typename Ret, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<WriterT<W, _M, Ret>(Args...)> >
    mbind(WriterT<W, _M, fdecay<Arg> > const& m, function_t<WriterT<W, _M, Ret>(Arg, Args...)> const& k){
        return invoke_f0(_([m, k](Args... args){
            return _do2(pa, m.run(), pb, k(fst(pa), args...).run(),
                return Monad<_M>::mreturn(make_pair_t(fst(pb), Monoid_t<W>::mappend(snd(pa), snd(pb)))););
        }));
    }
};

// MonadPlus
/*
instance (Monoid w, MonadPlus m) => MonadPlus (WriterT w m) where
    mzero       = WriterT mzero
    m `mplus` n = WriterT $ runWriterT m `mplus` runWriterT n
*/
template<typename W, class _M>
struct _is_monad_plus<_WriterT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> && _is_monad_plus<_M>::value> {};

template<typename W, class _M>
struct MonadPlus<_WriterT<W, _M> > : Monad<_WriterT<W, _M> >, _MonadPlus<_WriterT<W, _M> >
{
    using super = Monad<_WriterT<W, _M> >;

    // mzero = WriterT mzero
    template<typename A>
    static constexpr WriterT<W, _M, A> mzero(){
        return MonadPlus<_M>::template mzero<pair_t<A, W> >();
    }

    // m `mplus` n = WriterT $ runWriterT m `mplus` runWriterT n
    template<typename A>
    static constexpr WriterT<W, _M, A> mplus(WriterT<W, _M, A> const& m, WriterT<W, _M, A> const& n){
        return MonadPlus<_M>::mplus(m.run(), n.run());
    }
};

// Alternative
/*
instance (Monoid w, Alternative m) => Alternative (WriterT w m) where
    empty   = WriterT empty
    {-# INLINE empty #-}
    m <|> n = WriterT $ runWriterT m <|> runWriterT n
    {-# INLINE (<|>) #-}
*/
template<typename W, class _M>
struct _is_alternative<_WriterT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> && _is_alternative<_M>::value> {};

template<typename W, class _M>
struct Alternative<_WriterT<W, _M> > : _Alternative<_WriterT<W, _M> >
{
    // empty = WriterT empty
    template<typename A>
    static constexpr WriterT<W, _M, A> empty(){
        return Alternative<_M>::template empty<pair_t<A, W> >();
    }

    // m <|> n = WriterT $ runWriterT m <|> runWriterT n
    template<typename A>
    static constexpr WriterT<W, _M, A> alt_op(WriterT<W, _M, A> const& m, WriterT<W, _M, A> const& n){
        return m.run() | n.run();
    }
};

// Foldable
/*
instance (Foldable f) => Foldable (WriterT w f) where
    foldMap f = foldMap (f . fst) . runWriterT
    {-# INLINE foldMap #-}
#if MIN_VERSION_base(4,8,0)
    null (WriterT t) = null t
    length (WriterT t) = length t
#endif
*/
template<typename W, class _M>
struct _is_foldable<_WriterT<W, _M> > : _is_foldable<_M> {};

template<typename W, class _M>
struct Foldable<_WriterT<W, _M> > : _Foldable<_WriterT<W, _M> >
{
    // foldMap :: Monoid m => (a -> m) -> t a -> m
    // foldMap f = foldMap (f . fst) . runWriterT
    template<typename M1, typename Arg>
    static constexpr monoid_type<M1> foldMap(function_t<M1(Arg)> const& f, WriterT<W, _M, fdecay<Arg> > const& x){
        return Foldable<_M>::foldMap(f & _(fst<fdecay<Arg>, W>), x.run());
    }
};

// Traversable
/*
instance (Traversable f) => Traversable (WriterT w f) where
    traverse f = fmap WriterT . traverse f' . runWriterT where
       f' (a, b) = fmap (\ c -> (c, b)) (f a)
    {-# INLINE traverse #-}
*/
template<typename W, class _M>
struct _is_traversable<_WriterT<W, _M> > : _is_traversable<_M> {};

template<typename W, class _M>
struct Traversable<_WriterT<W, _M> > : _Traversable<_WriterT<W, _M> >
{
    // traverse :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
    // traverse f = fmap WriterT . traverse f' . runWriterT where
    //   f' (a, b) = fmap (\ c -> (c, b)) (f a)
    template<typename AP, typename Arg>
    static constexpr applicative_type<AP, typeof_t<AP, WriterT<W, _M, fdecay<Arg> > > >
    traverse(function_t<AP(Arg)> const& f, WriterT<W, _M, fdecay<Arg> > const& x)
    {
        using A = fdecay<Arg>;
        auto const f_ = _([&f](pair_t<A, W> const& p){
            return fmap(_([&p](A const& c){ return pair_t<A, W>(c, snd(p)); }), f(fst(p)));
        });
        return fmap(_(WriterT_<W, _M, A>), Traversable<_M>::traverse(f_, x.run()));
    }
};

// MonadZip
template<typename W, class _M>
struct MonadZip<_WriterT<W, _M> > : _MonadZip<MonadZip<_WriterT<W, _M> > >
{
    static_assert(is_monoid_v<W>, "Should be a Monoid");

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f (WriterT x) (WriterT y) = WriterT $
    //   mzipWith (\ ~(a, w) ~(b, w') -> (f a b, w `mappend` w')) x y
    template<typename C, typename AArg, typename BArg>
    static constexpr WriterT<W, _M, C> mzipWith(function_t<C(AArg, BArg)> const& f,
        WriterT<W, _M, fdecay<AArg> > const& x, WriterT<W, _M, fdecay<BArg> > const& y)
    {
        return MonadZip<_M>::mzipWith(_([&f](pair_t<fdecay<AArg>, W> const& pa, pair_t<fdecay<BArg>, W> const& pb){
            return pair_t<C, W>(f(fst(pa), fst(pb)), Monoid_t<W>::mappend(snd(pa), snd(pb)));
        }), x.run(), y.run());
    }
};

// MonadTrans
template<typename W>
struct MonadTrans<__WriterT<W> >
{
    static_assert(is_monoid_v<W>, "Should be a Monoid");

    // lift m = WriterT $ do
    //   a <- m
    //   return (a, mempty)
    template<typename M>
    static constexpr monad_type<M, WriterT<W, base_class_t<M>, value_type_t<M> > > lift(M const& m){
        return _do(a, m, return Monad_t<M>::mreturn(make_pair_t(a, W::mempty())););
    }
};

// MonadReader
template<typename R, typename W, class _M>
struct MonadReader<R, _WriterT<W, _M> > : _MonadReader<R, _WriterT<W, _M>, MonadReader<R, _WriterT<W, _M> > >
{
    using base_class = _WriterT<W, _M>;
    using super = _MonadReader<R, base_class, MonadReader<R, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // ask = lift ask
    static constexpr type<R> ask(){
        return MonadTrans<__WriterT<W> >::lift(MonadReader<R, _M>::ask());
    }

    // local = mapWriterT . local
    template<typename A, typename RArg>
    static constexpr std::enable_if_t<is_same_as_v<R, RArg>, type<A> >
    local(function_t<R(RArg)> const& f, type<A> const& m){
        return (_(mapWriterT<typename _M::template type<A>, typename _M::template type<A>>) & _(MonadReader<R, _M>::template local<A, RArg>))(f, m);
    }

    // reader = lift . reader
    template<typename A, typename RArg>
    static constexpr std::enable_if_t<is_same_as_v<R, RArg>, type<A> >
    reader(function_t<A(RArg)> const& f){
        return MonadTrans<__WriterT<W> >::lift(MonadReader<R, _M>::reader(f));
    }
};

// MonadWriter
template<typename W, class _M>
struct MonadWriter<W, _WriterT<W, _M> > : _MonadWriter<W, _WriterT<W, _M>, MonadWriter<W, _WriterT<W, _M> > >
{
    using base_class = _WriterT<W, _M>;
    using super = _MonadWriter<W, base_class, MonadWriter<W, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // writer = Lazy.writer
    template<typename A>
    static constexpr type<A> writer(pair_t<A, W> const& p){
        return _FUNCPROG::writer<_M>(p);
    }

    // tell = Lazy.tell
    static constexpr type<None> tell(W const& w){
        return _WriterT<W, _M>::tell(w);
    }

    // listen = Lazy.listen
    template<typename A>
    static constexpr type<pair_t<A, W> > listen(WriterT<W, _M, A> const& m){
        return m.listen();
    }

    // pass = Lazy.pass
    template<typename A>
    static constexpr type<A> pass(WriterT<W, _M, pair_t<A, function_t<W(W const&)> > > const& m){
        return _WriterT<W, _M>::pass(m);
    }
};

// MonadState
template<typename S, typename W, typename _M>
struct MonadState<S, _WriterT<W, _M> > : _MonadState<S, _WriterT<W, _M>, MonadState<S, _WriterT<W, _M> > >
{
    using base_class = _WriterT<W, _M>;
    using super = _MonadState<S, base_class, MonadState>;

    template<typename T>
    using type = typename super::template type<T>;

    // state = lift . state
    template<typename A>
    static constexpr type<A> state(function_t<pair_t<A, S>(S const&)> const& f){
        return MonadTrans<__WriterT<W> >::lift(MonadState<S, _M>::state(f));
    }

    // get = lift get
    static constexpr type<S> get(){
        return MonadTrans<__WriterT<W> >::lift(MonadState<S, _M>::get());
    }

    // put = lift . put
    static constexpr type<None> put(S const& s){
        return MonadTrans<__WriterT<W> >::lift(MonadState<S, _M>::put(s));
    }
};

_FUNCPROG_END

namespace std {

    template<typename W, class _M, typename A>
    ostream& operator<<(ostream& os, _FUNCPROG::WriterT<W, _M, A> const& x){
        return os <<  "WriterT[" << x.run() << ']';
    }

}
