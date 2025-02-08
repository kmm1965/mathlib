#pragma once

#include "../Monad/MonadReader.hpp"

_FUNCPROG_BEGIN

// MonadReader
//instance (MonadReader r m) => MonadReader r (ParsecT s u m) where
//    ask = lift ask
//    local f p = mkPT $ \s -> local f (runParsecT p s)

template<typename R, typename S, typename U, typename _M>
struct MonadReader<R, parsec::_ParsecT<S, U, _M> > :
    _MonadReader<R, parsec::_ParsecT<S, U, _M>, MonadReader<R, parsec::_ParsecT<S, U, _M> > >
{
    using base_class = parsec::_ParsecT<S, U, _M>;
    using super = _MonadReader<R, base_class, MonadReader<R, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // ask = lift ask
    static constexpr type<R> ask(){
        return MonadTrans<parsec::__ParsecT<S, U> >::lift(MonadReader<R, _M>::ask());
    }

    // local f p = mkPT $ \s -> local f (runParsecT p s)
    template<typename A, typename RArg>
    static constexpr typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    local(function_t<R(RArg)> const& f, type<A> const& p){
        return mkPT([=](parsec::State<S, U> const& s){ return MonadReader<R, _M>::local(f, parsec::runParsecT(p, s)); });
    }

    // reader = lift . reader
    template<typename A, typename RArg>
    static constexpr typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    reader(function_t<A(RArg)> const& f){
        return MonadTrans<parsec::__ParsecT<S, U> >::lift(MonadReader<R, _M>::reader(f));
    }
};

_FUNCPROG_END
