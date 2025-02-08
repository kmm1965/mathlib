#pragma once

#include "fwd/MonadPlus_fwd.hpp"
#include "fwd/Alternative_fwd.hpp"

_FUNCPROG_BEGIN

/*
-- -----------------------------------------------------------------------------
-- The MonadPlus class definition

-- | Monads that also support choice and failure.
class (Alternative m, Monad m) => MonadPlus m where
   -- | The identity of 'mplus'.  It should also satisfy the equations
   --
   -- > mzero >>= f  =  mzero
   -- > v >> mzero   =  mzero
   --
   -- The default definition is
   --
   -- @
   -- mzero = 'empty'
   -- @
   mzero :: m a
   mzero = empty

   -- | An associative operation. The default definition is
   --
   -- @
   -- mplus = ('<|>')
   -- @
   mplus :: m a -> m a -> m a
   mplus = (<|>)
*/
template<typename _MP>
struct _MonadPlus // Default implementation of some functions
{
    template<typename A>
    static constexpr alternative_type<typeof_t<_MP, A> > mzero();

    template<typename A>
    static constexpr alternative_type<typeof_t<_MP, A> >
    mplus(typeof_t<_MP, A> const& x, typeof_t<_MP, A> const& y);
};

_FUNCPROG_END
