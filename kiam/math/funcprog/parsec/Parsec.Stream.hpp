#pragma once

#include "../Maybe.hpp"
#include "../List.hpp"

_PARSEC_BEGIN

/*
-- | An instance of @Stream@ has stream type @s@, underlying monad @m@ and token type @t@ determined by the stream
--
-- Some rough guidelines for a \"correct\" instance of Stream:
--
--    * unfoldM uncons gives the [t] corresponding to the stream
--
--    * A @Stream@ instance is responsible for maintaining the \"position within the stream\" in the stream state @s@.  This is trivial unless you are using the monad in a non-trivial way.

class (Monad m) => Stream s m t | s -> t where
    uncons :: s -> m (Maybe (t,s))

instance (Monad m) => Stream [tok] m tok where
    uncons []     = return $ Nothing
    uncons (t:ts) = return $ Just (t,ts)
*/
template<typename _M, typename T>
struct Stream
{
    static_assert(is_monad<_M>::value, "_M should be a monad");

    using monad_type = _M;
    using token_type = T;
    using stream_type = List<token_type>;
    using pair_type = pair_t<token_type, stream_type>;
    using return_type = typename _M::template type<Maybe<pair_type> >;

    static constexpr return_type uncons(stream_type const& l) {
        return Monad_t<_M>::mreturn(null(l) ? Nothing<pair_type>() : Just(pair_type(head(l), tail(l))));
    }
};

_PARSEC_END
