#pragma once

#include "../Alternative.hpp"

_FUNCPROG_BEGIN

/*
-- -----------------------------------------------------------------------------
-- The Alternative class definition

infixl 3 <|>

-- | A monoid on applicative functors.
--
-- If defined, 'some' and 'many' should be the least solutions
-- of the equations:
--
-- * @'some' v = (:) '<$>' v '<*>' 'many' v@
--
-- * @'many' v = 'some' v '<|>' 'pure' []@
class Applicative f => Alternative f where
    -- | The identity of '<|>'
    empty :: f a
    -- | An associative binary operation
    (<|>) :: f a -> f a -> f a

    -- | One or more.
    some :: f a -> f [a]
    some v = some_v
      where
        many_v = some_v <|> pure []
        some_v = liftA2 (:) v many_v

    -- | Zero or more.
    many :: f a -> f [a]
    many v = many_v
      where
        many_v = some_v <|> pure []
        some_v = liftA2 (:) v many_v
*/
#define DECLARE_ALTERNATIVE_CLASS(ALT) \
    template<typename T> static constexpr ALT<T> empty(); \
    template<typename T> static constexpr ALT<T> alt_op(ALT<T> const& op1, ALT<T> const& op2);

template<class ALT>
constexpr alternative_type<ALT> operator|(ALT const& op1, ALT const& op2) {
    return Alternative_t<ALT>::alt_op(op1, op2);
}

_FUNCPROG_END
