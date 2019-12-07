#pragma once

#include "Parsec.Monad.hpp"
#include "Parsec.Alternative.hpp"

_FUNCPROG_BEGIN

// MonadPlus
template<typename S, typename U, typename _M>
struct is_monad_plus<parsec::_ParsecT<S, U, _M> > : std::true_type {};

template<typename S, typename U, typename _M>
struct is_same_monad_plus<parsec::_ParsecT<S, U, _M>, parsec::_ParsecT<S, U, _M> > : std::true_type {};

template<typename S, typename U, typename M>
struct MonadPlus<parsec::_ParsecT<S, U, M> > : Monad<parsec::_ParsecT<S, U, M> >, Alternative<parsec::_ParsecT<S, U, M> >
{
	template<typename A>
	parsec::ParsecT<S, U, M, A, parsec::parserZero_unParser<S, U, M, A> > mzero() {
        return Alternative<parsec::_ParsecT<S, U, M> >::template empty<A>();
    }
};

_FUNCPROG_END
