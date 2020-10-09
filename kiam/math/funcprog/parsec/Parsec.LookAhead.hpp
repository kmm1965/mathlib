#pragma once

_PARSEC_BEGIN

/*
-- | @lookAhead p@ parses @p@ without consuming any input.
--
-- If @p@ fails and consumes some input, so does @lookAhead@. Combine with 'try'
-- if this is undesirable.

lookAhead :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m a
lookAhead p =
    ParsecT $ \s _ cerr eok eerr -> do
        let eok' a _ _ = eok a s (newErrorUnknown (statePos s))
        unParser p s eok' cerr eok' eerr
*/
template<typename S, typename U, typename _M, typename A, typename P>
struct lookAhead_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    DECLARE_OK_ERR_TYPES();

    lookAhead_unParser(ParsecT<S, U, _M, A, P> const& p) : p(p){}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(State<S, U> const& s,
        ok_type<B> const& cok_, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        ok_type<B> const eok_ = [s, eok](A const& a, State<S, U> const&, ParseError const&){
            return eok(a, s, newErrorUnknown(statePos(s)));
        };
        return p.template run<B>(s, eok_, cerr, eok_, eerr);
    }

private:
    const ParsecT<S, U, _M, A, P> p;
};

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, A, lookAhead_unParser<S, U, _M, A, P> >
lookAhead(ParsecT<S, U, _M, A, P> const& p){
    return ParsecT<S, U, _M, A, lookAhead_unParser<S, U, _M, A, P> >(lookAhead_unParser<S, U, _M, A, P>(p));
}

_PARSEC_END
