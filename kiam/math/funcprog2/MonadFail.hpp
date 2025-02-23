#pragma once

#include "funcprog2_setup.h"

_FUNCPROG2_BEGIN

/*
-- instance Monad Foo where
--   (>>=) = {- ...bind impl... -}
--
--   -- Provide legacy 'fail' implementation for when
--   -- new-style MonadFail desugaring is not enabled.
--   fail = Fail.fail
--
-- instance Fail.MonadFail Foo where
--   fail = {- ...fail implementation... -}
-- @
--
-- See <https://prime.haskell.org/wiki/Libraries/Proposals/MonadFail>
-- for more details.
--
-- @since 4.9.0.0
--
module Control.Monad.Fail ( MonadFail(fail) ) where

import GHC.Base (String, Monad(), Maybe(Nothing), IO(), failIO)

-- | When a value is bound in @do@-notation, the pattern on the left
-- hand side of @<-@ might not match. In this case, this class
-- provides a function to recover.
--
-- A 'Monad' without a 'MonadFail' instance may only be used in conjunction
-- with pattern that always match, such as newtypes, tuples, data types with
-- only a single data constructor, and irrefutable patterns (@~pat@).
--
-- Instances of 'MonadFail' should satisfy the following law: @fail s@ should
-- be a left zero for 'Control.Monad.>>=',
--
-- @
-- fail s >>= f  =  fail s
-- @
--
-- If your 'Monad' is also 'Control.Monad.MonadPlus', a popular definition is
--
-- @
-- fail _ = mzero
-- @
--
-- @since 4.9.0.0
class Monad m => MonadFail m where
    fail :: String -> m a


-- | @since 4.9.0.0
instance MonadFail Maybe where
    fail _ = Nothing

-- | @since 4.9.0.0
instance MonadFail [] where
    {-# INLINE fail #-}
    fail _ = []

-- | @since 4.9.0.0
instance MonadFail IO where
    fail = failI
*/

// requires fail
template<typename _MF>
struct MonadFail;

_FUNCPROG2_END
