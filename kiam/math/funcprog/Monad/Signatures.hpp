#pragma once

#include "../funcprog_setup.h"

_FUNCPROG_BEGIN

/*
-- | Signature of the @callCC@ operation,
-- introduced in "Control.Monad.Trans.Cont".
-- Any lifting function @liftCallCC@ should satisfy
--
-- * @'lift' (f k) = f' ('lift' . k) => 'lift' (cf f) = liftCallCC cf f'@
--
type CallCC m a b = ((a -> m b) -> m a) -> m a
*/
template<typename MA, typename MB, typename Arg>
using CallCC = function_t<MA(function_t<MA(function_t<MB(Arg)> const&)> const&)>;

/*
-- | Signature of the @catchE@ operation,
-- introduced in "Control.Monad.Trans.Except".
-- Any lifting function @liftCatch@ should satisfy
--
-- * @'lift' (cf m f) = liftCatch ('lift' . cf) ('lift' f)@
--
type Catch e m a = m a -> (e -> m a) -> m a
*/
template<typename MA, typename EArg>
using Catch = function_t<MA(MA const&, function_t<MA(EArg)> const&)>;

/*
-- | Signature of the @listen@ operation,
-- introduced in "Control.Monad.Trans.Writer".
-- Any lifting function @liftListen@ should satisfy
--
-- * @'lift' . liftListen = liftListen . 'lift'@
--
type Listen w m a = m a -> m (a, w)
*/
template<typename W, typename _M, typename A>
using Listen = function_t<typename _M::template type<pair_t<A, W> >(typename _M::template type<A> const&)>;

/*
-- | Signature of the @pass@ operation,
-- introduced in "Control.Monad.Trans.Writer".
-- Any lifting function @liftPass@ should satisfy
--
-- * @'lift' . liftPass = liftPass . 'lift'@
--
type Pass w m a =  m (a, w -> w) -> m a
*/
template<typename W, typename _M, typename A>
using Pass = function_t<typename _M::template type<A>(typename _M::template type<pair_t<A, function_t<W(W const&)> > > const&)>;

_FUNCPROG_END
