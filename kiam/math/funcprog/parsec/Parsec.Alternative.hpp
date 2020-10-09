#pragma once

_PARSEC_BEGIN

/*
-- | @parserZero@ always fails without consuming any input. @parserZero@ is defined
-- equal to the 'mzero' member of the 'MonadPlus' class and to the 'Control.Applicative.empty' member
-- of the 'Control.Applicative.Alternative' class.

parserZero :: ParsecT s u m a
parserZero
    = ParsecT $ \s _ _ _ eerr ->
      eerr $ unknownError s
*/
template<typename S, typename U, typename _M, typename A>
struct parserZero_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    IMPLEMENT_UNPARSER_RUN(return eerr(unknownError(s));)
};

template<typename S, typename U, typename _M, typename A>
ParsecT<S, U, _M, A, parserZero_unParser<S, U, _M, A> > parserZero(){
    return ParsecT<S, U, _M, A, parserZero_unParser<S, U, _M, A> >(parserZero_unParser<S, U, _M, A>());
}

/*
parserPlus :: ParsecT s u m a -> ParsecT s u m a -> ParsecT s u m a
{-# INLINE parserPlus #-}
parserPlus m n
    = ParsecT $ \s cok cerr eok eerr ->
      let
          meerr err =
              let
                  neok y s' err' = eok y s' (mergeError err err')
                  neerr err' = eerr $ mergeError err err'
              in unParser n s cok cerr neok neerr
      in unParser m s cok cerr eok meerr
*/
template<typename S, typename U, typename _M, typename A, typename PM, typename PN>
struct parserPlus_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    DECLARE_OK_ERR_TYPES();

    parserPlus_unParser(ParsecT<S, U, _M, A, PM> const& m, ParsecT<S, U, _M, A, PN> const& n) : m(m), n(n){}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(State<S, U> const& s,
        ok_type<B> const& cok, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        err_type<B> const meerr =
            [&](ParseError const& err)
        {
            ok_type<B> const neok = [&eok, &err](A const& y, State<S, U> const& s_, ParseError const& err_){
                return eok(y, s_, mergeError(err, err_));
            };
            err_type<B> const neerr = [&eerr, &err](ParseError const& err_){
                return eerr(mergeError(err, err_));
            };
            return n.template run<B>(s, cok, cerr, neok, neerr);
        };
        return m.template run<B>(s, cok, cerr, eok, meerr);
    }

private:
    const ParsecT<S, U, _M, A, PM> m;
    const ParsecT<S, U, _M, A, PN> n;
};

DEFINE_FUNCTION_2(6, PARSECT(T0, T1, T2, T3, PARSERPLUS_UNPARSER(T0, T1, T2, T3, T4, T5)),
    parserPlus, PARSECT(T0, T1, T2, T3, T4) const&, m, PARSECT(T0, T1, T2, T3, T5) const&, n,
    return PARSECT(T0, T1, T2, T3, PARSERPLUS_UNPARSER(T0, T1, T2, T3, T4, T5))(PARSERPLUS_UNPARSER(T0, T1, T2, T3, T4, T5)(m, n));)

_PARSEC_END

_FUNCPROG_BEGIN

// Alternative
template<typename S, typename U, typename _M>
struct _is_alternative<parsec::_ParsecT<S, U, _M> > : _is_alternative<_M> {};

/*
-- | This combinator implements choice. The parser @p \<|> q@ first
-- applies @p@. If it succeeds, the value of @p@ is returned. If @p@
-- fails /without consuming any input/, parser @q@ is tried. This
-- combinator is defined equal to the 'mplus' member of the 'MonadPlus'
-- class and the ('Control.Applicative.<|>') member of 'Control.Applicative.Alternative'.
--
-- The parser is called /predictive/ since @q@ is only tried when
-- parser @p@ didn't consume any input (i.e.. the look ahead is 1).
-- This non-backtracking behaviour allows for both an efficient
-- implementation of the parser combinators and the generation of good
-- error messages.

(<|>) :: (ParsecT s u m a) -> (ParsecT s u m a) -> (ParsecT s u m a)
p1 <|> p2 = mplus p1 p2
*/

template<typename S, typename U, typename _M>
struct Alternative<parsec::_ParsecT<S, U, _M> >
{
    template<typename A>
    parsec::ParsecT<S, U, _M, A, parsec::parserZero_unParser<S, U, _M, A> > empty(){
        return parsec::parserZero<S, U, _M, A>();
    }
    
    template<typename T, typename P>
    struct alt_op_result_type;

    template<typename T, typename P>
    using alt_op_result_type_t = typename alt_op_result_type<T, P>::type;

    template<typename A, typename P, typename P2>
    struct alt_op_result_type<parsec::ParsecT<S, U, _M, A, P2>, P> {
        typedef parsec::ParsecT<S, U, _M, A, parsec::parserPlus_unParser<S, U, _M, A, P, P2> > type;
    };

    template<typename A, typename P, typename P2>
    static alt_op_result_type_t<parsec::ParsecT<S, U, _M, A, P2>, P>
    alt_op(parsec::ParsecT<S, U, _M, A, P> const& op1, parsec::ParsecT<S, U, _M, A, P2> const& op2){
        return parsec::parserPlus<S, U, _M, A, P, P2>(op1, op2);
    }
};

template<typename S, typename U, typename _M, typename A, typename P, typename P2>
using parsec_alt_op_result_type = typename Alternative<parsec::_ParsecT<S, U, _M> >::template alt_op_result_type_t<parsec::ParsecT<S, U, _M, A, P2>, P>;

template<typename S, typename U, typename _M, typename A, typename P, typename P2>
parsec_alt_op_result_type<S, U, _M, A, P, P2> operator|(parsec::ParsecT<S, U, _M, A, P> const& op1, parsec::ParsecT<S, U, _M, A, P2> const& op2){
    return Alternative<parsec::_ParsecT<S, U, _M> >::alt_op(op1, op2);
}

_FUNCPROG_END
