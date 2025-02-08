#pragma once

#include "../funcprog_setup.h"

_FUNCPROG_BEGIN

/*
-- ---------------------------------------------------------------------------
-- MonadWriter class
--
-- tell is like tell on the MUD's it shouts to monad
-- what you want to be heard. The monad carries this 'packet'
-- upwards, merging it if needed (hence the Monoid requirement).
--
-- listen listens to a monad acting, and returns what the monad "said".
--
-- pass lets you provide a writer transformer which changes internals of
-- the written object.

class (Monoid w, Monad m) => MonadWriter w m | m -> w where
*/
template<typename W, class _M>
struct MonadWriter;

template<typename W, typename M>
using MonadWriter_t = MonadWriter<W, base_class_t<M> >;

#define MONADWRITER_T_(W, M) BOOST_IDENTITY_TYPE((MonadWriter_t<W, M>))
#define MONADWRITER_T(W, M) typename MONADWRITER_T_(W, M)

template<typename W, class _M, typename MW>
struct _MonadWriter
{
    template<typename T>
    using type = typename _M::template type<T>;

    static_assert(is_monoid_v<W>, "Should be a Monoid");
    static_assert(_is_monad_v<_M>, "Should be a Monad");

/*
    -- | @'writer' (a,w)@ embeds a simple writer action.
    writer :: (a,w) -> m a
    writer ~(a, w) = do
      tell w
      return a
*/
    template<typename A>
    static constexpr type<A> writer(pair_t<A, W> const& p) {
        return MW::tell(snd(p)) >> Monad<_M>::mreturn(fst(p));
    }
/*
    -- | @'tell' w@ is an action that produces the output @w@.
    tell   :: w -> m ()
    tell w = writer ((),w)
*/
    static constexpr type<None> tell(W const& w) {
        return MW::writer(pair_t<None, W>(None(), w));
    }
/*
    -- | @'listen' m@ is an action that executes the action @m@ and adds
    -- its output to the value of the computation.
    listen :: m a -> m (a, w)

    -- | @'pass' m@ is an action that executes the action @m@, which
    -- returns a value and a function, and returns the value, applying
    -- the function to the output.
    pass   :: m (a, w -> w) -> m a
*/
};

/*
-- | @'listens' f m@ is an action that executes the action @m@ and adds
-- the result of applying @f@ to the output to the value of the computation.
--
-- * @'listens' f m = 'liftM' (id *** f) ('listen' m)@
listens :: MonadWriter w m => (w -> b) -> m a -> m (a, b)
listens f m = do
    ~(a, w) <- listen m
    return (a, f w)
*/
template<typename W, class M, typename B>
using listens_type = std::enable_if_t<
    is_monoid_v<W> && is_monad_v<M>,
    typename M::template type<pair_t<value_type_t<M>, B> >
>;

#define LISTENS_TYPE_(W, M, B) BOOST_IDENTITY_TYPE((listens_type<W, M, B>))
#define LISTENS_TYPE(W, M, B) typename LISTENS_TYPE_(W, M, B)

DECLARE_FUNCTION_2(3, LISTENS_TYPE(T0, T1, T2), listens, function_t<T2(T0 const&)> const&, T1 const&)
FUNCTION_TEMPLATE(3) constexpr LISTENS_TYPE(T0, T1, T2) listens(function_t<T2(T0 const&)> const& f, T1 const& m) {
    return _do(p, MONADWRITER_T_(T0, T1)::listen(m), return Monad_t<T1>::mreturn(pair_t<value_type_t<T1>, T2>(fst(p), f(snd(p)))););
}

/*
-- | @'censor' f m@ is an action that executes the action @m@ and
-- applies the function @f@ to its output, leaving the return value
-- unchanged.
--
-- * @'censor' f m = 'pass' ('liftM' (\\x -> (x,f)) m)@
censor :: MonadWriter w m => (w -> w) -> m a -> m a
censor f m = pass $ do
    a <- m
    return (a, f)
*/
template<typename W, class M>
using censor_type = std::enable_if_t<is_monoid_v<W> && is_monad_v<M>, M>;

#define CENSOR_TYPE_(W, M) BOOST_IDENTITY_TYPE((censor_type<W, M>))
#define CENSOR_TYPE(W, M) typename CENSOR_TYPE_(W, M)

DECLARE_FUNCTION_2(2, CENSOR_TYPE(T0, T1), censor, function_t<T0(T0 const&)> const&, T1 const&)
FUNCTION_TEMPLATE(2) constexpr CENSOR_TYPE(T0, T1) censor(function_t<T0(T0 const&)> const& f, T1 const& m) {
    return MONADWRITER_T_(T0, T1)::pass(
        _do(a, m, return Monad_t<T1>::mreturn(PAIR_T(value_type_t<T1>, function_t<T0(T0 const&)>)(a, f));)
    );
}

_FUNCPROG_END
