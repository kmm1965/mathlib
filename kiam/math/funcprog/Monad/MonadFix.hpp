#pragma once

#include "../func_traits.hpp"
#include "../y_combinator.hpp"

_FUNCPROG_BEGIN

/*
-- | Monads having fixed points with a \'knot-tying\' semantics.
-- Instances of 'MonadFix' should satisfy the following laws:
--
-- [/purity/]
--      @'mfix' ('return' . h)  =  'return' ('fix' h)@
--
-- [/left shrinking/ (or /tightening/)]
--      @'mfix' (\\x -> a >>= \\y -> f x y)  =  a >>= \\y -> 'mfix' (\\x -> f x y)@
--
-- [/sliding/]
--      @'mfix' ('Control.Monad.liftM' h . f)  =  'Control.Monad.liftM' h ('mfix' (f . h))@,
--      for strict @h@.
--
-- [/nesting/]
--      @'mfix' (\\x -> 'mfix' (\\y -> f x y))  =  'mfix' (\\x -> f x x)@
--
-- This class is used in the translation of the recursive @do@ notation
-- supported by GHC and Hugs.
class (Monad m) => MonadFix m where
        -- | The fixed point of a monadic computation.
        -- @'mfix' f@ executes the action @f@ only once, with the eventual
        -- output fed back as the input.  Hence @f@ should not be strict,
        -- for then @'mfix' f@ would diverge.
        mfix :: (a -> m a) -> m a
*/
template<typename T>
struct MonadFix;

template<typename T>
using MonadFix_t = MonadFix<base_class_t<T> >;

template<class Fun>
function_t<Fun> fix(Fun &&f)
{
    using A = first_argument_type_t<function_t<Fun> >;
    const y_combinator_result<Fun> comb = y_combinator(f);
    return [comb](A const& a) {
        return comb(a);
    };
}

_FUNCPROG_END
