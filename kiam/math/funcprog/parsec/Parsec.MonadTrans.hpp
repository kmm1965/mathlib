#pragma once

#include "../Monad/Trans/MonadTrans.hpp"

_FUNCPROG_BEGIN

// MonadTrans
//instance MonadTrans (ParsecT s u) where
//  lift amb = ParsecT $ \s _ _ eok _ -> do
//    a <- amb
//    eok a s $ unknownError s
template<typename S, typename U>
struct MonadTrans<parsec::__ParsecT<S, U> >
{
    template<typename M>
    struct lift_unParser
    {
        using ParsecT_base_t = parsec::ParsecT_base<S, U, base_class_t<M>, value_type_t<M> >;

        lift_unParser(M const& m) : m(m) {}

        IMPLEMENT_UNPARSER_RUN(return _do(a, m, return eok(a, s, parsec::unknownError(s)););)

    private:
        M const m;
    };

    template<typename M>
    static constexpr monad_type<M, parsec::ParsecT<S, U, base_class_t<M>, value_type_t<M>, void> > lift(M const& m){
        return lift_unParser<M>(m);
    }
};

_FUNCPROG_END
