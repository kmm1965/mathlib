#pragma once

#include "Parsec.Monad.hpp"
#include "Parsec.Alternative.hpp"

_FUNCPROG_BEGIN

// MonadPlus
template<typename S, typename U, typename _M>
struct _is_monad_plus<parsec::_ParsecT<S, U, _M> > : std::true_type {};

template<typename S, typename U, typename _M>
struct MonadPlus<parsec::_ParsecT<S, U, _M> > :
    Monad<parsec::_ParsecT<S, U, _M> >, Alternative<parsec::_ParsecT<S, U, _M> >
{
    template<typename A>
    parsec::ParsecT<S, U, _M, A, parsec::parserZero_unParser<S, U, _M, A> > mzero(){
        return Alternative<parsec::_ParsecT<S, U, _M> >::template empty<A>();
    }
};

_FUNCPROG_END
